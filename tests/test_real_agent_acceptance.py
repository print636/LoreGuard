from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import Settings
from app.provider import ModelResult, ProviderCallTelemetry, ProviderError
from app.review_agent import AGENT_SYSTEM_PROMPT
import scripts.run_real_agent_acceptance as real_runner
from scripts.run_agent_acceptance import DEFAULT_SUITE_ROOT, load_manifest
from scripts.run_real_agent_acceptance import (
    PILOT_TASK_IDS,
    EVALUATION_IMPLEMENTATION_FILES,
    SyntheticExtractionProvider,
    execute_production_task,
    main as real_runner_main,
    run_real_acceptance,
    sanitize_execution_task,
    score_execution,
    score_suite,
)


MANIFEST_PATH = DEFAULT_SUITE_ROOT / "manifest.json"
CANARY_KEY = "s" + "k-test-must-never-appear"
ROTATED_CANARY_KEY = "s" + "k-rotated-test-must-never-appear"
CANARY_ENDPOINT = "https://credentials-must-never-appear.invalid/v1"


def recorded_settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key=CANARY_KEY,
        openai_base_url=CANARY_ENDPOINT,
        openai_model="recorded-agent-test-model",
        provider_thinking_mode="disabled",
        enable_model_extraction=True,
        enable_review_agent=True,
        provider_max_attempts=1,
        per_run_token_budget=50_000,
        review_agent_token_budget=12_000,
        review_agent_total_deadline_seconds=30,
    )


def _telemetry(
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
    *,
    category: str = "success",
):
    return ProviderCallTelemetry(
        input_chars=128,
        response_chars=96,
        received_bytes=96,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        elapsed_ms=3,
        category=category,
        attempt_no=1,
        http_status=200 if category == "success" else None,
    )


class RecordedAgentProvider:
    """A no-network protocol recording used only to exercise the production graph."""

    def __init__(
        self,
        task: dict,
        *,
        settings: Settings | None = None,
        behavior: str | None = None,
    ):
        self.settings = settings or recorded_settings()
        self.task_id = task["id"]
        self.scenario = behavior or (
            "read_then_abstain"
            if task["scenario"] == "failed_patch_then_abstain"
            else task["scenario"]
        )
        self.candidate = task["initial_candidate"]
        self.read_span = task["allowed_evidence"][0]
        oracle = task["oracle"]
        self.patch = oracle.get("expected_patch", oracle.get("attempt_patch"))
        self.calls = 0
        self.forks = 0
        self.prompt_pairs: list[tuple[str, str]] = []

    @property
    def configured(self):
        return True

    def fork_for_agent(self, bounded_settings):
        self.settings = bounded_settings
        self.forks += 1
        return self

    def complete(self, system, user):
        self.calls += 1
        self.prompt_pairs.append((system, user))
        if self.scenario == "direct_abstain" or (
            self.scenario == "read_then_abstain" and self.calls > 1
        ):
            payload = {
                "actions": [
                    {
                        "action": "ABSTAIN",
                        "candidate_indexes": [1],
                        "reason_code": "insufficient_evidence",
                    }
                ]
            }
        elif self.calls == 1:
            payload = {
                "actions": [
                    {
                        "action": "READ_SPAN",
                        "requests": [
                            {
                                "candidate_index": 1,
                                "doc_ref": self.read_span["doc_ref"],
                                "line_start": self.read_span["line_start"],
                                "line_end": self.read_span["line_end"],
                            }
                        ],
                    }
                ]
            }
        else:
            observations = json.loads(user)["tool_observations"]
            span_id = next(
                row["span_id"] for row in observations if row.get("tool") == "READ_SPAN"
            )
            payload = {
                "actions": [
                    {
                        "action": "PATCH_RECORDS",
                        "patches": [
                            {
                                "candidate_index": 1,
                                "doc_ref": self.candidate["doc_ref"],
                                "span_id": span_id,
                                "fields": self.patch,
                            }
                        ],
                    }
                ]
            }
        return ModelResult(
            text=json.dumps(payload, ensure_ascii=False),
            prompt_tokens=11,
            completion_tokens=7,
            telemetry=_telemetry(),
        )


class RejectingMainDelegate:
    def __init__(self):
        self.settings = recorded_settings()
        self.complete_calls = 0
        self.forks = 0

    def complete(self, _system, _user):
        self.complete_calls += 1
        raise AssertionError("main extraction reached the Agent delegate")

    def fork_for_agent(self, bounded_settings):
        self.settings = bounded_settings
        self.forks += 1
        return self


class RecordedRuntimeFailureProvider(RecordedAgentProvider):
    def complete(self, system, user):
        self.calls += 1
        self.prompt_pairs.append((system, user))
        telemetry = _telemetry(
            prompt_tokens=0,
            completion_tokens=0,
            category="read_timeout",
        )
        raise ProviderError(
            "recorded safe timeout",
            category="read_timeout",
            telemetry=telemetry,
        )


class RealAgentAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest(DEFAULT_SUITE_ROOT)
        cls.tasks = {task["id"]: task for task in cls.manifest["tasks"]}

    def test_execution_task_is_a_positive_allowlist_without_oracle(self):
        task = self.tasks[PILOT_TASK_IDS[0]]
        safe = sanitize_execution_task(task)
        self.assertEqual(
            {
                "id",
                "persona",
                "documents",
                "initial_candidate",
                "validator_reason",
            },
            set(safe),
        )
        serialized = json.dumps(safe, ensure_ascii=False)
        self.assertNotIn("oracle", serialized)
        self.assertNotIn("allowed_evidence", serialized)
        self.assertNotIn("expected_patch", serialized)
        self.assertNotIn("attempt_patch", serialized)

    def test_runner_defaults_to_v2_and_keeps_explicit_v1_selection(self):
        current = real_runner._parser().parse_args(["--execute"])
        legacy = real_runner._parser().parse_args(
            ["--execute", "--benchmark-version", "v1"]
        )
        diagnostic = real_runner._parser().parse_args(
            [
                "--execute",
                "--agent-timeout-seconds",
                "30",
                "--agent-total-deadline-seconds",
                "60",
            ]
        )
        self.assertEqual("v2", current.benchmark_version)
        self.assertEqual("v1", legacy.benchmark_version)
        self.assertIsNone(current.agent_timeout_seconds)
        self.assertIsNone(current.agent_total_deadline_seconds)
        self.assertEqual(30.0, diagnostic.agent_timeout_seconds)
        self.assertEqual(60.0, diagnostic.agent_total_deadline_seconds)

    def test_real_report_boundary_resanitizes_protocol_diagnostics(self):
        marker = "DO_NOT_PERSIST_MODEL_OUTPUT"
        oversized = (marker + ".") * 10_000
        safe = real_runner._safe_agent_run(
            {
                "trace": [
                    {
                        "action": "DECISION",
                        "protocol_diagnostic": {
                            "stage": [marker, {"nested": marker}],
                            "root_shape": {"nested": [marker]},
                            "action_count": 999,
                            "action_names": [
                                [marker],
                                {"nested": marker},
                                "ABSTAIN",
                            ],
                            "schema_error_locations": [
                                oversized,
                                [marker],
                                {"nested": marker},
                            ],
                            "schema_error_types": [
                                [marker],
                                {"nested": marker},
                                "extra_forbidden",
                            ],
                        },
                    }
                ]
            }
        )
        serialized = json.dumps(safe, ensure_ascii=False)
        self.assertNotIn(marker, serialized)
        diagnostic = safe["trace"][0]["protocol_diagnostic"]
        self.assertEqual("action_schema", diagnostic["stage"])
        self.assertEqual("unparsed", diagnostic["root_shape"])
        self.assertEqual(7, diagnostic["action_count"])
        self.assertEqual(
            ["unknown", "unknown", "ABSTAIN"], diagnostic["action_names"]
        )
        self.assertEqual(
            [
                ".".join(["unknown_field"] * 6),
                "unknown_field",
                "unknown_field",
            ],
            diagnostic["schema_error_locations"],
        )
        self.assertEqual(
            ["validation_error", "validation_error", "extra_forbidden"],
            diagnostic["schema_error_types"],
        )
        self.assertLess(len(json.dumps(diagnostic, ensure_ascii=False)), 1_000)

    def test_real_report_boundary_bounds_all_trace_line_numbers(self):
        huge = real_runner.MAX_SAFE_AGENT_LINE_NUMBER + 1
        safe = real_runner._safe_agent_run(
            {
                "trace": [
                    {
                        "action": "READ_SPAN",
                        "line_start": huge,
                        "line_end": -1,
                        "allowed_line_start": True,
                        "allowed_line_end": huge,
                    }
                ]
            }
        )["trace"][0]
        for field in (
            "line_start",
            "line_end",
            "allowed_line_start",
            "allowed_line_end",
        ):
            self.assertIsNone(safe[field])

    def test_evaluation_timeout_overrides_are_explicit_and_bounded(self):
        for arguments, message in (
            (["--agent-timeout-seconds", "31"], "agent-timeout-seconds"),
            (
                ["--agent-total-deadline-seconds", "61"],
                "agent-total-deadline-seconds",
            ),
            (
                [
                    "--agent-timeout-seconds",
                    "30",
                    "--agent-total-deadline-seconds",
                    "20",
                ],
                "must be >=",
            ),
        ):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(
                SystemExit, message
            ):
                real_runner_main(["--execute", *arguments])

    def test_scorer_fails_closed_on_invalid_frozen_read_configuration(self):
        selected = [self.tasks[PILOT_TASK_IDS[0]]]
        invalid_values = (
            ("agent_context_radius", True),
            ("agent_context_radius", "20"),
            ("agent_context_radius", -1),
            ("agent_context_radius", 51),
            ("agent_max_read_lines", False),
            ("agent_max_read_lines", "12"),
            ("agent_max_read_lines", -1),
            ("agent_max_read_lines", 0),
            ("agent_max_read_lines", 21),
        )
        for key, value in invalid_values:
            with self.subTest(key=key, value=value), self.assertRaisesRegex(
                ValueError, f"invalid frozen evaluation configuration: {key}"
            ):
                score_suite(
                    self.manifest,
                    selected,
                    [],
                    repeats=1,
                    suite_mode="pilot",
                    suite_root=DEFAULT_SUITE_ROOT,
                    evaluation_configuration={key: value},
                )

        defaults = score_suite(
            self.manifest,
            selected,
            [],
            repeats=1,
            suite_mode="pilot",
            suite_root=DEFAULT_SUITE_ROOT,
            evaluation_configuration={},
        )["evaluation_coverage"]
        self.assertEqual(
            real_runner.DEFAULT_REVIEW_AGENT_CONTEXT_RADIUS_LINES,
            defaults["configured_agent_context_radius_lines"],
        )
        self.assertEqual(
            real_runner.DEFAULT_REVIEW_AGENT_MAX_READ_LINES,
            defaults["configured_agent_max_read_lines"],
        )

        strict = score_suite(
            self.manifest,
            selected,
            [],
            repeats=1,
            suite_mode="pilot",
            suite_root=DEFAULT_SUITE_ROOT,
            evaluation_configuration={
                "agent_context_radius": 0,
                "agent_max_read_lines": 20,
            },
        )["evaluation_coverage"]
        self.assertEqual(0, strict["configured_agent_context_radius_lines"])
        self.assertEqual(20, strict["configured_agent_max_read_lines"])

    def test_agent_prompt_targets_lexical_core_fields_not_label_only_patches(self):
        self.assertIn("只因一个或多个核心字段缺少原文词面支持", AGENT_SYSTEM_PROMPT)
        self.assertIn('"fields":{"location":"原文中的地点词组"}', AGENT_SYSTEM_PROMPT)
        self.assertIn("event 的 time/location/participants", AGENT_SYSTEM_PROMPT)
        self.assertIn("禁止修改 modality", AGENT_SYSTEM_PROMPT)
        self.assertIn("它们不能修复 lexical_support", AGENT_SYSTEM_PROMPT)
        self.assertIn("所问对象或语义身份换成另一个问题", AGENT_SYSTEM_PROMPT)
        self.assertIn("第一轮可以直接 ABSTAIN", AGENT_SYSTEM_PROMPT)
        self.assertNotIn("词面支持或语义标签校验", AGENT_SYSTEM_PROMPT)

    def test_synthetic_provider_never_delegates_main_extraction(self):
        task = self.tasks[PILOT_TASK_IDS[0]]
        delegate = RejectingMainDelegate()
        wrapper = SyntheticExtractionProvider(
            delegate,
            task["initial_candidate"],
            document_chars=100,
        )
        result = wrapper.complete("not-stored", "not-stored")
        self.assertIn('"records"', result.text)
        self.assertEqual(0, delegate.complete_calls)
        captured = wrapper.fork_for_agent(wrapper.settings)
        self.assertIsNot(captured, delegate)
        self.assertEqual(delegate.settings, captured.settings)
        self.assertEqual(1, delegate.forks)

    def test_final_provider_messages_exclude_scorer_answers_and_other_documents(self):
        task = self.tasks["gp-09-cross-branch-merge"]
        provider = RecordedAgentProvider(task)
        artifact = execute_production_task(
            sanitize_execution_task(task),
            provider,
            suite_root=DEFAULT_SUITE_ROOT,
        )
        self.assertEqual(2, len(provider.prompt_pairs))
        forbidden_markers = (
            task["id"],
            task["persona"],
            '"oracle"',
            '"allowed_evidence"',
            '"expected_patch"',
            '"attempt_patch"',
        )
        for system, user in provider.prompt_pairs:
            for marker in forbidden_markers:
                self.assertNotIn(marker, system)
                self.assertNotIn(marker, user)
            self.assertNotIn("B 路线", system)
            self.assertNotIn("旧港仓库盘点物资", user)
        first_user = provider.prompt_pairs[0][1]
        second_user = provider.prompt_pairs[1][1]
        self.assertNotIn("洛岚在灯塔顶层查看潮位", first_user)
        self.assertIn("洛岚在灯塔顶层查看潮位", second_user)
        first_payload = json.loads(first_user)
        second_payload = json.loads(second_user)
        literal_fields = second_payload["tool_observations"][0][
            "literal_fields_already_present"
        ]
        self.assertTrue(set(literal_fields).issubset(
            first_payload["candidates"][0]["patch_field_allowlist"]
        ))
        self.assertTrue(all(isinstance(field, str) for field in literal_fields))
        self.assertNotIn(
            "literal_fields_already_present",
            json.dumps(artifact, ensure_ascii=False),
        )

    def test_recorded_pilot_drives_production_path_but_cannot_claim_model_quality(self):
        artifacts = []
        observed_paths = set()
        for task_id in PILOT_TASK_IDS:
            task = self.tasks[task_id]
            artifact = execute_production_task(
                sanitize_execution_task(task),
                RecordedAgentProvider(task),
                suite_root=DEFAULT_SUITE_ROOT,
            )
            score = score_execution(task, artifact, suite_root=DEFAULT_SUITE_ROOT)
            self.assertTrue(score["correct"], (task_id, score))
            self.assertTrue(score["production_valid"])
            self.assertTrue(score["trace_replayable"])
            self.assertTrue(score["telemetry_complete"])
            self.assertTrue(score["surface_patch_match"])
            observed_paths.add(tuple(score["actual_path"]))
            artifacts.append(artifact)
        self.assertEqual(2, len(observed_paths))
        report = score_suite(
            self.manifest,
            [self.tasks[task_id] for task_id in PILOT_TASK_IDS],
            artifacts,
            repeats=1,
            suite_mode="pilot",
            suite_root=DEFAULT_SUITE_ROOT,
        )
        self.assertFalse(report["passed"])
        self.assertTrue(report["gates"]["production"])
        self.assertFalse(report["gates"]["real_provider_connected"])
        self.assertFalse(report["real_provider_connected"])
        self.assertFalse(report["eligible_for_model_quality_claims"])
        self.assertEqual("real-agent-acceptance-report-v2", report["schema_version"])
        self.assertEqual("development_tuned_pilot_only_not_holdout", report["claim_scope"])
        self.assertEqual(3, report["subgroups"]["development_tuned"]["task_count"])
        self.assertEqual(0, report["subgroups"]["holdout"]["task_count"])
        self.assertNotIn("at_least_three_dynamic_paths", report["gates"])
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn(CANARY_KEY, serialized)
        self.assertNotIn(CANARY_ENDPOINT, serialized)
        for task_id in PILOT_TASK_IDS:
            task = self.tasks[task_id]
            for patch in (
                task["oracle"].get("expected_patch"),
                task["oracle"].get("attempt_patch"),
            ):
                if patch:
                    for value in patch.values():
                        self.assertNotIn(str(value), serialized)
        for artifact in artifacts:
            self.assertNotIn("response_text", serialized)
            self.assertNotIn("raw_candidate", serialized)
            self.assertEqual(
                sum(
                    1
                    for event in artifact["agent_runs"][0]["trace"]
                    if event["action"] == "DECISION"
                ),
                len(artifact["agent_provider_calls"]),
            )
            for patch_hash in artifact["model_patch_field_sha256s"]:
                self.assertRegex(patch_hash, r"^[a-f0-9]{64}$")

    def test_unrecoverable_outcome_is_not_penalized_for_a_safer_path(self):
        cases = (
            ("gp-04-invented-route-event", "read_then_abstain"),
            ("gp-06-unknown-box-opener", "direct_abstain"),
        )
        for task_id, behavior in cases:
            with self.subTest(task_id=task_id, behavior=behavior):
                task = self.tasks[task_id]
                artifact = execute_production_task(
                    sanitize_execution_task(task),
                    RecordedAgentProvider(task, behavior=behavior),
                    suite_root=DEFAULT_SUITE_ROOT,
                )
                score = score_execution(task, artifact, suite_root=DEFAULT_SUITE_ROOT)
                self.assertFalse(score["reference_path_match"])
                self.assertTrue(score["outcome_match"])
                self.assertTrue(score["outcome_policy_match"])
                self.assertTrue(score["runtime_success"])
                self.assertTrue(score["semantic_abstain"])
                self.assertTrue(score["correct"], score)

    def test_runtime_failure_is_safe_degradation_but_not_semantic_abstain(self):
        task = self.tasks["ed-05-anonymous-letter"]
        artifact = execute_production_task(
            sanitize_execution_task(task),
            RecordedRuntimeFailureProvider(task),
            suite_root=DEFAULT_SUITE_ROOT,
        )
        score = score_execution(task, artifact, suite_root=DEFAULT_SUITE_ROOT)
        self.assertTrue(score["outcome_match"])
        self.assertTrue(score["trace_replayable"])
        self.assertEqual(0, score["accepted_safety_violations"])
        self.assertFalse(score["runtime_success"])
        self.assertFalse(score["semantic_abstain"])
        self.assertFalse(score["outcome_policy_match"])
        self.assertFalse(score["correct"])
        self.assertIn("provider:read_timeout", score["runtime_failure_categories"])
        self.assertIn("terminal:provider_error", score["runtime_failure_categories"])
        report = score_suite(
            self.manifest,
            [task],
            [artifact],
            repeats=1,
            suite_mode="pilot",
            suite_root=DEFAULT_SUITE_ROOT,
        )
        self.assertEqual(0.0, report["metrics"]["semantic_abstain_rate"])
        self.assertEqual(1, report["metrics"]["runtime_failure_count"])
        self.assertEqual(1, report["metrics"]["read_timeout_calls"])
        self.assertEqual([], report["metrics"]["dynamic_paths"])

    def test_rejected_patch_is_containment_and_quality_failure(self):
        task = self.tasks["ed-09-invalid-conditional-inverse"]
        artifact = execute_production_task(
            sanitize_execution_task(task),
            RecordedAgentProvider(task, behavior="failed_patch_then_abstain"),
            suite_root=DEFAULT_SUITE_ROOT,
        )
        score = score_execution(task, artifact, suite_root=DEFAULT_SUITE_ROOT)
        self.assertTrue(score["runtime_success"])
        self.assertFalse(score["semantic_abstain"])
        self.assertTrue(score["safe_containment_success"])
        self.assertTrue(score["bad_patch_proposal"])
        self.assertFalse(score["correct"])
        report = score_suite(
            self.manifest,
            [task],
            [artifact],
            repeats=1,
            suite_mode="pilot",
            suite_root=DEFAULT_SUITE_ROOT,
        )
        self.assertEqual(1.0, report["metrics"]["bad_patch_proposal_rate"])
        self.assertEqual(1, report["metrics"]["safe_containment_success_count"])
        self.assertEqual(0.0, report["metrics"]["semantic_abstain_rate"])
        self.assertEqual([], report["metrics"]["dynamic_paths"])

    def test_every_rejected_patch_reason_is_bad_containment_not_quality(self):
        task = self.tasks["ed-09-invalid-conditional-inverse"]
        artifact = execute_production_task(
            sanitize_execution_task(task),
            RecordedAgentProvider(task, behavior="failed_patch_then_abstain"),
            suite_root=DEFAULT_SUITE_ROOT,
        )
        for reason in (
            "patch_validation_failed",
            "semantic_promotion",
            "semantic_field_forbidden",
            "patch_field_forbidden",
        ):
            with self.subTest(reason=reason):
                attacked = copy.deepcopy(artifact)
                patch_event = next(
                    event
                    for event in attacked["agent_runs"][0]["trace"]
                    if event["action"] == "PATCH_RECORDS"
                )
                patch_event["validator_reason"] = reason
                score = score_execution(task, attacked, suite_root=DEFAULT_SUITE_ROOT)
                self.assertTrue(score["runtime_success"])
                self.assertTrue(score["bad_patch_proposal"])
                self.assertEqual([reason], score["bad_patch_rejection_reasons"])
                self.assertTrue(score["safe_containment_success"])
                self.assertFalse(score["semantic_abstain"])
                self.assertFalse(score["correct"])
                report = score_suite(
                    self.manifest,
                    [task],
                    [attacked],
                    repeats=1,
                    suite_mode="pilot",
                    suite_root=DEFAULT_SUITE_ROOT,
                )
                self.assertEqual(
                    {reason: 1},
                    report["metrics"]["bad_patch_rejection_reasons"],
                )

    def test_bad_patch_cannot_be_reclassified_as_semantic_abstention(self):
        task = self.tasks["ed-09-invalid-conditional-inverse"]
        artifact = execute_production_task(
            sanitize_execution_task(task),
            RecordedAgentProvider(task, behavior="failed_patch_then_abstain"),
            suite_root=DEFAULT_SUITE_ROOT,
        )
        attacked = copy.deepcopy(artifact)
        run = attacked["agent_runs"][0]
        final = run["trace"][-1]
        final.update(
            action="ABSTAIN",
            validator_reason="explicit_abstain",
            final="abstained",
        )
        run["final_reason"] = "explicit_abstain"
        score = score_execution(task, attacked, suite_root=DEFAULT_SUITE_ROOT)
        self.assertTrue(score["bad_patch_proposal"])
        self.assertFalse(score["semantic_abstain"])
        self.assertFalse(score["outcome_policy_match"])
        self.assertFalse(score["correct"])

    def test_accepted_patch_for_unrecoverable_task_is_bad_not_containment(self):
        task = self.tasks["ed-09-invalid-conditional-inverse"]
        artifact = execute_production_task(
            sanitize_execution_task(task),
            RecordedAgentProvider(task, behavior="failed_patch_then_abstain"),
            suite_root=DEFAULT_SUITE_ROOT,
        )
        attacked = copy.deepcopy(artifact)
        patch_event = next(
            event
            for event in attacked["agent_runs"][0]["trace"]
            if event["action"] == "PATCH_RECORDS"
        )
        patch_event.update(final="accepted", validator_reason="patch_ok")
        score = score_execution(task, attacked, suite_root=DEFAULT_SUITE_ROOT)
        self.assertTrue(score["bad_patch_proposal"])
        self.assertEqual(
            ["oracle_unrecoverable_patch"], score["bad_patch_rejection_reasons"]
        )
        self.assertFalse(score["safe_containment_success"])
        self.assertFalse(score["semantic_abstain"])
        self.assertFalse(score["correct"])

    def test_implementation_bundle_includes_pinned_runtime_dependencies(self):
        self.assertIn("requirements.txt", EVALUATION_IMPLEMENTATION_FILES)
        self.assertIn("app/review_agent.py", EVALUATION_IMPLEMENTATION_FILES)
        self.assertIn("app/model_extractor.py", EVALUATION_IMPLEMENTATION_FILES)
        self.assertIn("app/usage.py", EVALUATION_IMPLEMENTATION_FILES)
        self.assertIn("app/parser.py", EVALUATION_IMPLEMENTATION_FILES)
        self.assertIn("app/natural.py", EVALUATION_IMPLEMENTATION_FILES)
        self.assertIn("scripts/run_agent_acceptance.py", EVALUATION_IMPLEMENTATION_FILES)

    def test_full_report_reserves_27_tasks_for_holdout_gates(self):
        report = score_suite(
            self.manifest,
            self.manifest["tasks"],
            [],
            repeats=3,
            suite_mode="full",
            suite_root=DEFAULT_SUITE_ROOT,
        )
        self.assertEqual("full_with_separate_holdout_gate", report["claim_scope"])
        self.assertEqual(3, report["subgroups"]["development_tuned"]["task_count"])
        self.assertEqual(27, report["subgroups"]["holdout"]["task_count"])
        self.assertEqual(81, report["subgroups"]["holdout"]["expected_execution_count"])
        self.assertEqual(
            {persona: 1 for persona in self.manifest["personas"]},
            report["subgroups"]["development_tuned"]["task_persona_distribution"],
        )
        self.assertEqual(
            {persona: 9 for persona in self.manifest["personas"]},
            report["subgroups"]["holdout"]["task_persona_distribution"],
        )
        self.assertTrue(report["gates"]["holdout_task_count_27"])
        self.assertTrue(report["gates"]["manifest_persona_tasks_10_each"])
        self.assertTrue(report["gates"]["development_persona_tasks_1_each"])
        self.assertTrue(report["gates"]["holdout_persona_tasks_9_each"])
        self.assertFalse(report["gates"]["holdout_persona_executions_27_each"])
        self.assertFalse(report["gates"]["holdout_execution_coverage"])
        self.assertIn("at_least_three_runtime_success_semantic_paths", report["gates"])
        self.assertTrue(report["gates"]["full_repetitions_3"])
        one_repeat = score_suite(
            self.manifest,
            self.manifest["tasks"],
            [],
            repeats=1,
            suite_mode="full",
            suite_root=DEFAULT_SUITE_ROOT,
        )
        self.assertFalse(one_repeat["gates"]["full_repetitions_3"])
        self.assertFalse(one_repeat["gates"]["holdout_expected_executions_81"])

    def test_full_scripted_replay_proves_all_engineering_gates_not_model_quality(self):
        artifacts = []
        for task in self.manifest["tasks"]:
            for repeat in range(1, 4):
                artifacts.append(
                    execute_production_task(
                        sanitize_execution_task(task),
                        RecordedAgentProvider(task),
                        suite_root=DEFAULT_SUITE_ROOT,
                        repeat=repeat,
                    )
                )
        report = score_suite(
            self.manifest,
            self.manifest["tasks"],
            artifacts,
            repeats=3,
            suite_mode="full",
            suite_root=DEFAULT_SUITE_ROOT,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["gates"]["real_provider_connected"])
        self.assertFalse(report["eligible_for_model_quality_claims"])
        self.assertEqual(
            {persona: 27 for persona in self.manifest["personas"]},
            report["subgroups"]["holdout"]["execution_persona_distribution"],
        )
        for gate, passed in report["gates"].items():
            if gate != "real_provider_connected":
                self.assertTrue(passed, gate)

    def test_full_dynamic_gate_uses_three_successful_semantic_paths(self):
        cases = (
            ("gp-01-core-value-rewrite", None),
            ("gp-04-invented-route-event", "read_then_abstain"),
            ("gp-05-invented-exemption", "direct_abstain"),
        )
        artifacts = []
        for task_id, behavior in cases:
            task = self.tasks[task_id]
            artifacts.append(
                execute_production_task(
                    sanitize_execution_task(task),
                    RecordedAgentProvider(task, behavior=behavior),
                    suite_root=DEFAULT_SUITE_ROOT,
                )
            )
        report = score_suite(
            self.manifest,
            self.manifest["tasks"],
            artifacts,
            repeats=3,
            suite_mode="full",
            suite_root=DEFAULT_SUITE_ROOT,
        )
        self.assertEqual(
            ["ABSTAIN", "READ_SPAN -> ABSTAIN", "READ_SPAN -> PATCH_RECORDS"],
            report["metrics"]["dynamic_paths"],
        )
        self.assertTrue(
            report["gates"]["at_least_three_runtime_success_semantic_paths"]
        )
        self.assertTrue(report["gates"]["runtime_success_recover_path_present"])
        self.assertTrue(
            report["gates"]["runtime_success_direct_abstain_path_present"]
        )
        self.assertTrue(
            report["gates"]["runtime_success_semantic_abstain_path_present"]
        )
        self.assertEqual(
            report["metrics"]["dynamic_paths"],
            report["subgroups"]["holdout"]["dynamic_paths"],
        )

    def test_full_rejects_repeat_override_before_provider_creation(self):
        def forbidden_factory(_task, _repeat):
            raise AssertionError("provider must not be created for invalid full repeats")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "requires.*repeats 3"):
                run_real_acceptance(
                    manifest_path=MANIFEST_PATH,
                    suite_root=DEFAULT_SUITE_ROOT,
                    suite_mode="full",
                    repeats=1,
                    provider_factory=forbidden_factory,
                    checkpoint_path=root / "full.checkpoint.json",
                    report_path=root / "full.report.json",
                )
        with self.assertRaisesRegex(SystemExit, "requires.*repeats 3"):
            real_runner_main(
                [
                    "--execute",
                    "--suite",
                    "full",
                    "--repeats",
                    "1",
                    "--manifest",
                    str(MANIFEST_PATH),
                ]
            )

    def test_scorer_reanchors_candidate_and_span_hashes_to_frozen_inputs(self):
        task = self.tasks["gp-09-cross-branch-merge"]
        genuine = execute_production_task(
            sanitize_execution_task(task),
            RecordedAgentProvider(task),
            suite_root=DEFAULT_SUITE_ROOT,
        )
        self.assertTrue(score_execution(task, genuine)["correct"])
        for field, forged in (
            ("candidate_hash", "c" * 64),
            ("span_hash", "d" * 64),
        ):
            with self.subTest(field=field):
                attacked = copy.deepcopy(genuine)
                for event in attacked["agent_runs"][0]["trace"]:
                    if event["action"] in {"READ_SPAN", "PATCH_RECORDS"}:
                        event[field] = forged
                score = score_execution(task, attacked, suite_root=DEFAULT_SUITE_ROOT)
                self.assertFalse(score["correct"])
                self.assertFalse(score["trace_replayable"])
                self.assertGreater(score["accepted_safety_violations"], 0)
                self.assertIn(f"{field}_mismatch", score["trace_reasons"])

        attacked = copy.deepcopy(genuine)
        attacked["model_patch_field_sha256s"] = ["e" * 64]
        score = score_execution(task, attacked, suite_root=DEFAULT_SUITE_ROOT)
        self.assertTrue(score["fingerprint_match"])
        self.assertFalse(score["surface_patch_match"])
        self.assertTrue(score["bad_patch_proposal"])
        self.assertEqual(
            ["oracle_patch_mismatch"], score["bad_patch_rejection_reasons"]
        )
        self.assertFalse(score["safe_containment_success"])
        self.assertFalse(score["correct"])

    def test_read_covering_oracle_target_is_an_evidence_hit(self):
        task = self.tasks["gp-09-cross-branch-merge"]
        genuine = execute_production_task(
            sanitize_execution_task(task),
            RecordedAgentProvider(task),
            suite_root=DEFAULT_SUITE_ROOT,
        )
        expanded = copy.deepcopy(genuine)
        metadata = next(row for row in task["documents"] if row["doc_ref"] == "d1")
        lines = real_runner._resolve_document(
            DEFAULT_SUITE_ROOT, metadata["path"]
        ).read_text(encoding="utf-8").splitlines()
        expanded_hash = real_runner._sha256_text("\n".join(lines[0:2]))
        for event in expanded["agent_runs"][0]["trace"]:
            if event["action"] in {"READ_SPAN", "PATCH_RECORDS"}:
                event.update(line_start=1, line_end=2, span_hash=expanded_hash)
        score = score_execution(task, expanded, suite_root=DEFAULT_SUITE_ROOT)
        self.assertTrue(score["trace_replayable"], score)
        self.assertEqual([], score["trace_reasons"])
        self.assertEqual(0, score["accepted_safety_violations"])
        self.assertTrue(score["correct"])

    def test_oracle_target_is_not_reported_as_server_authorization(self):
        task = self.tasks["gp-09-cross-branch-merge"]
        genuine = execute_production_task(
            sanitize_execution_task(task),
            RecordedAgentProvider(task),
            suite_root=DEFAULT_SUITE_ROOT,
        )
        scorer_variant = copy.deepcopy(task)
        scorer_variant["allowed_evidence"] = [
            {"doc_ref": "d1", "line_start": 1, "line_end": 1}
        ]
        score = score_execution(
            scorer_variant, genuine, suite_root=DEFAULT_SUITE_ROOT
        )
        self.assertFalse(score["trace_replayable"])
        self.assertIn("read_misses_oracle_evidence", score["trace_reasons"])
        self.assertEqual(0, score["accepted_safety_violations"])
        self.assertFalse(score["correct"])

    def test_oracle_hit_does_not_excuse_read_outside_server_window(self):
        task = self.tasks["gp-09-cross-branch-merge"]
        genuine = execute_production_task(
            sanitize_execution_task(task),
            RecordedAgentProvider(task),
            suite_root=DEFAULT_SUITE_ROOT,
        )
        attacked = copy.deepcopy(genuine)
        metadata = next(row for row in task["documents"] if row["doc_ref"] == "d1")
        lines = real_runner._resolve_document(
            DEFAULT_SUITE_ROOT, metadata["path"]
        ).read_text(encoding="utf-8").splitlines()
        expanded_hash = real_runner._sha256_text("\n".join(lines[0:2]))
        for event in attacked["agent_runs"][0]["trace"]:
            if event["action"] in {"READ_SPAN", "PATCH_RECORDS"}:
                event.update(line_start=1, line_end=2, span_hash=expanded_hash)
        score = score_execution(
            task,
            attacked,
            suite_root=DEFAULT_SUITE_ROOT,
            server_context_radius_lines=0,
        )
        self.assertFalse(score["trace_replayable"])
        self.assertIn("read_outside_server_window", score["trace_reasons"])
        self.assertNotIn("read_misses_oracle_evidence", score["trace_reasons"])
        self.assertGreater(score["accepted_safety_violations"], 0)
        self.assertFalse(score["correct"])

    def test_accepted_read_must_respect_frozen_max_read_lines(self):
        task = self.tasks["gp-09-cross-branch-merge"]
        genuine = execute_production_task(
            sanitize_execution_task(task),
            RecordedAgentProvider(task),
            suite_root=DEFAULT_SUITE_ROOT,
        )
        attacked = copy.deepcopy(genuine)
        metadata = next(row for row in task["documents"] if row["doc_ref"] == "d1")
        lines = real_runner._resolve_document(
            DEFAULT_SUITE_ROOT, metadata["path"]
        ).read_text(encoding="utf-8").splitlines()
        expanded_hash = real_runner._sha256_text("\n".join(lines[0:2]))
        for event in attacked["agent_runs"][0]["trace"]:
            if event["action"] in {"READ_SPAN", "PATCH_RECORDS"}:
                event.update(line_start=1, line_end=2, span_hash=expanded_hash)
        score = score_execution(
            task,
            attacked,
            suite_root=DEFAULT_SUITE_ROOT,
            server_max_read_lines=1,
        )
        self.assertFalse(score["trace_replayable"])
        self.assertIn("read_outside_server_window", score["trace_reasons"])
        self.assertGreater(score["accepted_safety_violations"], 0)

    def test_read_trace_rejects_non_production_accepted_final(self):
        task = self.tasks["gp-09-cross-branch-merge"]
        genuine = execute_production_task(
            sanitize_execution_task(task),
            RecordedAgentProvider(task),
            suite_root=DEFAULT_SUITE_ROOT,
        )
        attacked = copy.deepcopy(genuine)
        read = next(
            row
            for row in attacked["agent_runs"][0]["trace"]
            if row["action"] == "READ_SPAN"
        )
        self.assertEqual("continue", read["final"])
        read["final"] = "accepted"
        score = score_execution(task, attacked, suite_root=DEFAULT_SUITE_ROOT)
        self.assertFalse(score["trace_replayable"])
        self.assertIn("invalid_read_final", score["trace_reasons"])

    def test_rejected_read_diagnostic_must_prove_the_frozen_range_violation(self):
        task = self.tasks["gp-09-cross-branch-merge"]
        genuine = execute_production_task(
            sanitize_execution_task(task),
            RecordedAgentProvider(task),
            suite_root=DEFAULT_SUITE_ROOT,
        )
        source_run = genuine["agent_runs"][0]
        decision = copy.deepcopy(
            next(row for row in source_run["trace"] if row["action"] == "DECISION")
        )
        source_read = copy.deepcopy(
            next(row for row in source_run["trace"] if row["action"] == "READ_SPAN")
        )

        def forged_rejection(*, start=2, end=2, allowed_start=1, allowed_end=2, span=None):
            artifact = copy.deepcopy(genuine)
            rejected = {
                **source_read,
                "line_start": start,
                "line_end": end,
                "allowed_line_start": allowed_start,
                "allowed_line_end": allowed_end,
                "span_hash": span,
                "validator_reason": "evidence_range",
                "final": "rejected",
            }
            final = {
                **source_read,
                "action": "FINALIZE",
                "round": 1,
                "line_start": 2,
                "line_end": 2,
                "allowed_line_start": None,
                "allowed_line_end": None,
                "span_hash": None,
                "validator_reason": "evidence_range",
                "final": "abstained",
            }
            run = copy.deepcopy(source_run)
            run.update(
                trace=[decision, rejected, final],
                total_trace_events=3,
                trace_truncated=False,
                recovered_records=0,
                unresolved_records=1,
                abstained_records=1,
                final_reason="evidence_range",
            )
            artifact["agent_runs"] = [run]
            artifact["model_directive_count"] = 0
            artifact["final_directive_fingerprint"] = None
            artifact["model_patch_field_sha256s"] = []
            return artifact

        no_violation = score_execution(
            task,
            forged_rejection(),
            suite_root=DEFAULT_SUITE_ROOT,
            server_max_read_lines=1,
        )
        self.assertFalse(no_violation["trace_replayable"])
        self.assertIn(
            "invalid_read_preflight_diagnostic", no_violation["trace_reasons"]
        )

        proven = score_execution(
            task,
            forged_rejection(start=1, end=2),
            suite_root=DEFAULT_SUITE_ROOT,
            server_max_read_lines=1,
        )
        self.assertTrue(proven["trace_replayable"], proven)

        for attacked in (
            forged_rejection(start=1, end=2, allowed_end=1),
            forged_rejection(start=1, end=2, span="f" * 64),
            forged_rejection(
                start=real_runner.MAX_SAFE_AGENT_LINE_NUMBER + 1,
                end=real_runner.MAX_SAFE_AGENT_LINE_NUMBER + 1,
            ),
        ):
            with self.subTest(attacked=attacked["agent_runs"][0]["trace"][1]):
                score = score_execution(
                    task,
                    attacked,
                    suite_root=DEFAULT_SUITE_ROOT,
                    server_max_read_lines=1,
                )
                self.assertFalse(score["trace_replayable"])
                self.assertIn(
                    "invalid_read_preflight_diagnostic", score["trace_reasons"]
                )

    def test_checkpoint_resume_skips_completed_provider_calls_and_rejects_drift(self):
        factory_calls = []
        active_settings = [recorded_settings()]

        def factory(execution_task, repeat):
            factory_calls.append((execution_task["id"], repeat))
            return RecordedAgentProvider(
                self.tasks[execution_task["id"]], settings=active_settings[0]
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "pilot.checkpoint.json"
            report_path = root / "pilot.report.json"
            report = run_real_acceptance(
                manifest_path=MANIFEST_PATH,
                suite_root=DEFAULT_SUITE_ROOT,
                suite_mode="pilot",
                repeats=1,
                provider_factory=factory,
                checkpoint_path=checkpoint,
                report_path=report_path,
            )
            self.assertFalse(report["passed"])
            self.assertTrue(report["gates"]["production"])
            self.assertFalse(report["gates"]["real_provider_connected"])
            self.assertEqual(3, len(factory_calls))
            self.assertEqual(
                15.0,
                report["evaluation_coverage"]["configured_agent_timeout_seconds"],
            )
            self.assertEqual(
                active_settings[0].review_agent_max_read_lines,
                report["evaluation_coverage"]["configured_agent_max_read_lines"],
            )
            self.assertEqual(
                active_settings[0].review_agent_context_radius_lines,
                report["evaluation_coverage"][
                    "configured_agent_context_radius_lines"
                ],
            )
            self.assertEqual(0, report["metrics"]["read_timeout_calls"])
            checkpoint_text = checkpoint.read_text(encoding="utf-8")
            checkpoint_payload = json.loads(checkpoint_text)
            configuration = checkpoint_payload["header"]["configuration"]
            self.assertEqual(
                "scripted_harness", checkpoint_payload["header"]["execution_source"]
            )
            previous = "0" * 64
            for execution in checkpoint_payload["executions"]:
                self.assertEqual(
                    previous, execution["previous_execution_chain_sha256"]
                )
                self.assertRegex(execution["execution_content_sha256"], r"^[a-f0-9]{64}$")
                self.assertRegex(execution["execution_chain_sha256"], r"^[a-f0-9]{64}$")
                previous = execution["execution_chain_sha256"]
            self.assertRegex(
                checkpoint_payload["header"]["agent_prompt_sha256"],
                r"^[a-f0-9]{64}$",
            )
            self.assertRegex(
                checkpoint_payload["header"]["evaluation_implementation_bundle_sha256"],
                r"^[a-f0-9]{64}$",
            )
            self.assertRegex(configuration["endpoint_identifier_sha256"], r"^[a-f0-9]{64}$")
            self.assertEqual(
                active_settings[0].per_run_token_budget,
                configuration["per_run_token_budget"],
            )
            self.assertEqual(
                active_settings[0].review_agent_max_read_lines,
                configuration["agent_max_read_lines"],
            )
            self.assertEqual(
                active_settings[0].review_agent_context_radius_lines,
                configuration["agent_context_radius"],
            )
            self.assertNotIn(CANARY_KEY, checkpoint_text)
            self.assertNotIn(CANARY_ENDPOINT, checkpoint_text)

            tampered_payload = copy.deepcopy(checkpoint_payload)
            tampered_payload["executions"][0]["status"][
                "review_agent_succeeded"
            ] = not bool(
                tampered_payload["executions"][0]["status"].get(
                    "review_agent_succeeded"
                )
            )
            tampered = root / "tampered.checkpoint.json"
            tampered.write_text(
                json.dumps(tampered_payload, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "content/chain drift"):
                run_real_acceptance(
                    manifest_path=MANIFEST_PATH,
                    suite_root=DEFAULT_SUITE_ROOT,
                    suite_mode="pilot",
                    repeats=1,
                    provider_factory=factory,
                    checkpoint_path=tampered,
                    report_path=root / "tampered.report.json",
                    resume=True,
                )
            factory_calls.clear()
            factory_calls.clear()

            # A credential rotation is deliberately not part of the frozen
            # configuration and must not invalidate a completed checkpoint.
            active_settings[0] = active_settings[0].model_copy(
                update={"openai_api_key": ROTATED_CANARY_KEY}
            )
            resumed = run_real_acceptance(
                manifest_path=MANIFEST_PATH,
                suite_root=DEFAULT_SUITE_ROOT,
                suite_mode="pilot",
                repeats=1,
                provider_factory=factory,
                checkpoint_path=checkpoint,
                report_path=report_path,
                resume=True,
            )
            self.assertFalse(resumed["passed"])
            self.assertFalse(resumed["eligible_for_model_quality_claims"])
            self.assertNotIn(ROTATED_CANARY_KEY, checkpoint.read_text(encoding="utf-8"))
            self.assertNotIn(ROTATED_CANARY_KEY, report_path.read_text(encoding="utf-8"))
            # One provider instance is created only to re-verify the frozen config.
            self.assertEqual([(PILOT_TASK_IDS[0], 1)], factory_calls)

            active_settings[0] = active_settings[0].model_copy(
                update={"openai_base_url": "https://different-gateway.invalid/v1"}
            )
            with self.assertRaisesRegex(ValueError, "drift"):
                run_real_acceptance(
                    manifest_path=MANIFEST_PATH,
                    suite_root=DEFAULT_SUITE_ROOT,
                    suite_mode="pilot",
                    repeats=1,
                    provider_factory=factory,
                    checkpoint_path=checkpoint,
                    report_path=report_path,
                    resume=True,
                )

            active_settings[0] = recorded_settings()
            with patch.object(
                real_runner,
                "AGENT_SYSTEM_PROMPT",
                AGENT_SYSTEM_PROMPT + " prompt-drift",
            ):
                with self.assertRaisesRegex(ValueError, "drift"):
                    run_real_acceptance(
                        manifest_path=MANIFEST_PATH,
                        suite_root=DEFAULT_SUITE_ROOT,
                        suite_mode="pilot",
                        repeats=1,
                        provider_factory=factory,
                        checkpoint_path=checkpoint,
                        report_path=report_path,
                        resume=True,
                    )

            with patch.object(
                real_runner,
                "_implementation_bundle_sha256",
                return_value="f" * 64,
            ):
                with self.assertRaisesRegex(ValueError, "drift"):
                    run_real_acceptance(
                        manifest_path=MANIFEST_PATH,
                        suite_root=DEFAULT_SUITE_ROOT,
                        suite_mode="pilot",
                        repeats=1,
                        provider_factory=factory,
                        checkpoint_path=checkpoint,
                        report_path=report_path,
                        resume=True,
                    )

            original_read_bytes = Path.read_bytes
            baseline_bundle = real_runner._implementation_bundle_sha256()
            runtime_files = ("app/usage.py", "app/parser.py", "app/natural.py")
            for relative_path in runtime_files:
                target = (real_runner.ROOT / relative_path).resolve()

                def read_with_drift(path, *, drift_target=target):
                    data = original_read_bytes(path)
                    return data + b"\n# simulated-drift" if path.resolve() == drift_target else data

                with patch.object(Path, "read_bytes", new=read_with_drift):
                    drifted_bundle = real_runner._implementation_bundle_sha256()
                self.assertNotEqual(baseline_bundle, drifted_bundle)
                with self.subTest(runtime_file=relative_path), patch.object(
                    real_runner,
                    "_implementation_bundle_sha256",
                    return_value=drifted_bundle,
                ):
                    with self.assertRaisesRegex(ValueError, "drift"):
                        run_real_acceptance(
                            manifest_path=MANIFEST_PATH,
                            suite_root=DEFAULT_SUITE_ROOT,
                            suite_mode="pilot",
                            repeats=1,
                            provider_factory=factory,
                            checkpoint_path=checkpoint,
                            report_path=report_path,
                            resume=True,
                        )

            active_settings[0] = recorded_settings().model_copy(
                update={"per_run_token_budget": 49_999}
            )
            with self.assertRaisesRegex(ValueError, "drift"):
                run_real_acceptance(
                    manifest_path=MANIFEST_PATH,
                    suite_root=DEFAULT_SUITE_ROOT,
                    suite_mode="pilot",
                    repeats=1,
                    provider_factory=factory,
                    checkpoint_path=checkpoint,
                    report_path=report_path,
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
