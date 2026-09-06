from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.provider import ModelResult, ProviderCallTelemetry
from scripts.run_agent_acceptance import DEFAULT_SUITE_ROOT, load_manifest
from scripts.run_real_agent_acceptance import (
    PILOT_TASK_IDS,
    SyntheticExtractionProvider,
    execute_production_task,
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


def _telemetry(prompt_tokens: int = 11, completion_tokens: int = 7):
    return ProviderCallTelemetry(
        input_chars=128,
        response_chars=96,
        received_bytes=96,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        elapsed_ms=3,
        category="success",
        attempt_no=1,
        http_status=200,
    )


class RecordedAgentProvider:
    """A no-network protocol recording used only to exercise the production graph."""

    def __init__(self, task: dict, *, settings: Settings | None = None):
        self.settings = settings or recorded_settings()
        self.task_id = task["id"]
        self.scenario = task["scenario"]
        self.candidate = task["initial_candidate"]
        oracle = task["oracle"]
        self.patch = oracle.get("expected_patch", oracle.get("attempt_patch"))
        self.calls = 0
        self.forks = 0

    @property
    def configured(self):
        return True

    def fork_for_agent(self, bounded_settings):
        self.settings = bounded_settings
        self.forks += 1
        return self

    def complete(self, _system, user):
        self.calls += 1
        if self.scenario == "direct_abstain":
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
                                "doc_ref": self.candidate["doc_ref"],
                                "line_start": self.candidate["source_line_start"],
                                "line_end": self.candidate["source_line_end"],
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
        self.assertIs(delegate, wrapper.fork_for_agent(wrapper.settings))
        self.assertEqual(1, delegate.forks)

    def test_three_recorded_paths_drive_the_full_production_agent(self):
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
            observed_paths.add(tuple(score["actual_path"]))
            artifacts.append(artifact)
        self.assertEqual(3, len(observed_paths))
        report = score_suite(
            self.manifest,
            [self.tasks[task_id] for task_id in PILOT_TASK_IDS],
            artifacts,
            repeats=1,
            suite_mode="pilot",
            suite_root=DEFAULT_SUITE_ROOT,
        )
        self.assertTrue(report["passed"], report["gates"])
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

    def test_scorer_reanchors_candidate_and_span_hashes_to_frozen_inputs(self):
        task = self.tasks["nw-01-location-literal"]
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
            self.assertTrue(report["passed"])
            self.assertEqual(3, len(factory_calls))
            checkpoint_text = checkpoint.read_text(encoding="utf-8")
            checkpoint_payload = json.loads(checkpoint_text)
            configuration = checkpoint_payload["header"]["configuration"]
            self.assertRegex(configuration["endpoint_identifier_sha256"], r"^[a-f0-9]{64}$")
            self.assertEqual(
                active_settings[0].per_run_token_budget,
                configuration["per_run_token_budget"],
            )
            self.assertNotIn(CANARY_KEY, checkpoint_text)
            self.assertNotIn(CANARY_ENDPOINT, checkpoint_text)
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
            self.assertTrue(resumed["passed"])
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
