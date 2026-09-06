from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.embeddings import EmbeddingProfile
from app.evidence_chunks import EvidenceChunker, SnapshotDocumentKey
from app.evidence_rag import (
    EvidenceRetrievalResult,
    RankedEvidence,
    RetrievalDiagnostics,
)
from scripts.run_evidence_retrieval_eval import (
    DEFAULT_SUITE_ROOT,
    PINNED_FREEZE_SHA256,
    PINNED_MANIFEST_SHA256,
    RetrievalRunnerError,
    RunnerConfiguration,
    _assert_clean_tracked_worktree,
    _assert_distinct_paths,
    _aggregate_task_scores,
    _canonical_bytes,
    _write_json,
    compare_score_reports,
    main,
    prepare_suite,
    run_prepared,
    score_prediction_files,
    validate_prepared,
    validate_predictions,
    validate_score_report,
)


class _FakeExecutor:
    def __init__(
        self,
        prepared: dict,
        configuration: RunnerConfiguration,
        profile: EmbeddingProfile,
    ) -> None:
        self._prepared = prepared
        self._configuration = configuration
        self._profile = profile
        self._chunker = EvidenceChunker(
            target_chars=configuration.chunk_target_chars,
            min_chars=configuration.chunk_min_chars,
            max_chars=configuration.chunk_max_chars,
            overlap_chars=configuration.chunk_overlap_chars,
        )
        empty_stats = {
            "expected_chunks": 1,
            "reused_chunks": 0,
            "embedded_chunks": 0,
            "provider_calls": 0,
            "provider_input_chars": 0,
            "elapsed_ms": 0,
            "complete_documents": 0,
            "incomplete_documents": 0,
            "failure_reasons": {},
        }
        reused_stats = {**empty_stats, "reused_chunks": 1}
        wrong_profile = EmbeddingProfile.openai_compatible(
            provider_namespace="evidence-retrieval-v1-fixture",
            model_identifier="fixture-only/wrong-space",
            model_revision="fixture-v0",
            deployment_fingerprint="frozen-wrong-profile-768-v0",
            document_transform_identity="fixture-vector-v0",
            query_transform_identity="fixture-vector-v0",
            dimensions=768,
        )
        visible = len(prepared["documents"])
        self.metadata = {
            "chunker_version": self._chunker.version,
            "cold": empty_stats,
            "reuse": reused_stats,
            "corpus": {
                "visible_documents": visible,
                "active_authorizable_documents": sum(
                    row["active"] and row["role"] in {"canon", "chapter"}
                    for row in prepared["documents"]
                ),
                "inactive_versions": sum(not row["active"] for row in prepared["documents"]),
                "decoy_documents": sum(
                    row["role"] == "isolation_decoy" for row in prepared["documents"]
                ),
                "projects": len({row["project_id"] for row in prepared["documents"]}),
            },
            "provider": {
                "used_for_index": False,
                "cold_complete": True,
                "reuse_complete": True,
            },
            "isolation_fixtures": {
                "wrong_profile_vectors": 1,
                "wrong_profile_logical_id": "distractor-wrong-profile-768-v0",
                "wrong_profile_canonical_id": wrong_profile.profile_id,
                "wrong_profile_dimensions": 768,
                "wrong_profile_on_authorized_chunk": True,
                "old_versions_and_decoys_loaded": 3,
            },
            "pg_storage_bytes": 0,
        }

    def retrieve(self, task: dict, repeat_index: int) -> EvidenceRetrievalResult:
        del repeat_index
        target = task["allowed_snapshots"][0]
        document = next(
            row
            for row in self._prepared["documents"]
            if row["project_id"] == target["project_id"]
            and row["document_id"] == target["document_id"]
            and row["version"] == target["version"]
        )
        chunks = self._chunker.chunk(
            project_id=document["project_id"],
            document_id=document["document_id"],
            document_version=document["version"],
            content=document["content"],
            content_sha256=document["content_sha256"],
        )
        match = RankedEvidence(
            chunk=chunks[0],
            rrf_score=1 / 61,
            keyword_rank=1,
            vector_rank=None,
            entity_rank=None,
        )
        return EvidenceRetrievalResult(
            matches=(match,),
            diagnostics=RetrievalDiagnostics(
                strategy="keyword-only",
                mode="lexical_only",
                reason=None,
                profile_id=self._profile.profile_id,
                chunker_version=self._chunker.version,
                candidate_count=len(chunks),
                keyword_hits=1,
                vector_hits=0,
                entity_hits=0,
                result_count=1,
                provider_calls=0,
                provider_input_chars=0,
                elapsed_ms=1,
            ),
        )


class EvidenceRetrievalEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prepared = prepare_suite(split="dev")
        self.configuration = RunnerConfiguration(strategy="keyword-only")
        self.profile = EmbeddingProfile.openai_compatible(
            provider_namespace="eval-test",
            model_identifier="BAAI/bge-small-zh-v1.5",
            model_revision="7999e1d3359715c523056ef9478215996d62a620",
            deployment_fingerprint="unit-test",
            dimensions=512,
        )

    def _prediction_fixture(self) -> tuple[dict, dict]:
        prepared = self.prepared
        predictions = run_prepared(
            prepared,
            prepared_sha256=hashlib.sha256(_canonical_bytes(prepared)).hexdigest(),
            configuration=self.configuration,
            executor=_FakeExecutor(prepared, self.configuration, self.profile),
            canonical_profile=self.profile,
            code_commit="a" * 40,
        )
        return prepared, predictions

    def test_prepare_is_oracle_free_but_keeps_old_and_decoy_documents(self):
        serialized = json.dumps(self.prepared, ensure_ascii=False)
        for forbidden in (
            "expected_evidence",
            "labels",
            "negative_candidates",
            "difficulty",
        ):
            self.assertNotIn(f'"{forbidden}"', serialized)
        self.assertTrue(any(not row["active"] for row in self.prepared["documents"]))
        self.assertTrue(
            any(row["role"] == "isolation_decoy" for row in self.prepared["documents"])
        )
        for task in self.prepared["tasks"]:
            self.assertTrue(task["allowed_snapshots"])
            self.assertTrue(
                all(
                    next(
                        row
                        for row in self.prepared["documents"]
                        if row["project_id"] == snapshot["project_id"]
                        and row["document_id"] == snapshot["document_id"]
                        and row["version"] == snapshot["version"]
                    )["active"]
                    for snapshot in task["allowed_snapshots"]
                )
            )

    def test_run_input_schema_forbids_extra_fields_at_every_level(self):
        cases = []
        top = json.loads(json.dumps(self.prepared))
        top["oracle"] = True
        cases.append(top)
        nested = json.loads(json.dumps(self.prepared))
        nested["tasks"][0]["labels"] = ["leak"]
        cases.append(nested)
        deep = json.loads(json.dumps(self.prepared))
        deep["tasks"][0]["allowed_snapshots"][0]["unexpected"] = "leak"
        cases.append(deep)
        for value in cases:
            with self.subTest(keys=value.keys()):
                with self.assertRaises(RetrievalRunnerError):
                    validate_prepared(value)

    def test_run_requires_execute_before_reading_files_or_connecting(self):
        with self.assertRaisesRegex(RetrievalRunnerError, "explicit --execute"):
            main(
                [
                    "run",
                    "--prepared",
                    "missing.json",
                    "--predictions",
                    "unused.json",
                    "--database-url",
                    "postgresql+psycopg://invalid",
                    "--strategy",
                    "keyword-only",
                    "--embedding-base-url",
                    "http://127.0.0.1:1/v1",
                    "--embedding-model",
                    "fixture",
                    "--embedding-revision",
                    "fixture",
                    "--embedding-deployment-fingerprint",
                    "fixture",
                    "--embedding-dimensions",
                    "512",
                ]
            )

    def test_run_is_stable_and_predictions_contain_no_query_or_text(self):
        prepared_sha = hashlib.sha256(_canonical_bytes(self.prepared)).hexdigest()
        predictions = run_prepared(
            self.prepared,
            prepared_sha256=prepared_sha,
            configuration=self.configuration,
            executor=_FakeExecutor(self.prepared, self.configuration, self.profile),
            canonical_profile=self.profile,
            code_commit="a" * 40,
        )
        row = predictions["tasks"][0]
        self.assertEqual(3, len(row["rankings"]))
        self.assertEqual(row["rankings"][0], row["rankings"][1])
        self.assertEqual(row["rankings"][1], row["rankings"][2])
        payload = json.dumps(predictions, ensure_ascii=False)
        self.assertNotIn('"query"', payload)
        self.assertNotIn('"content"', payload)
        self.assertNotIn('"text"', payload)
        self.assertEqual(
            self.prepared["logical_profile"]["profile_id"],
            predictions["profile"]["logical_profile_id"],
        )
        self.assertEqual(self.profile.profile_id, predictions["profile"]["canonical_profile_id"])

    def test_score_verifies_prediction_hash_before_opening_oracle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path = root / "prepared.json"
            predictions_path = root / "predictions.json"
            _write_json(prepared_path, self.prepared)
            predictions_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RetrievalRunnerError, "hash does not match"):
                score_prediction_files(
                    prepared_path=prepared_path,
                    predictions_path=predictions_path,
                    expected_prediction_sha256="0" * 64,
                    manifest_path=root / "oracle-must-not-be-opened.json",
                )

    def test_score_uses_full_line_coverage_and_reports_latency(self):
        executor = _FakeExecutor(self.prepared, self.configuration, self.profile)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path = root / "prepared.json"
            predictions_path = root / "predictions.json"
            prepared_sha = _write_json(prepared_path, self.prepared)
            predictions = run_prepared(
                self.prepared,
                prepared_sha256=prepared_sha,
                configuration=self.configuration,
                executor=executor,
                canonical_profile=self.profile,
                code_commit="b" * 40,
            )
            prediction_sha = _write_json(predictions_path, predictions)
            report = score_prediction_files(
                prepared_path=prepared_path,
                predictions_path=predictions_path,
                expected_prediction_sha256=prediction_sha,
                manifest_path=DEFAULT_SUITE_ROOT / "manifest.json",
            )
        metrics = report["metrics"]
        self.assertEqual(16, metrics["task_count"])
        self.assertEqual(0, metrics["isolation_failures"])
        self.assertEqual(48, metrics["latency_ms"]["samples"])
        self.assertEqual(report, validate_score_report(report))
        serialized = json.dumps(report, ensure_ascii=False)
        for forbidden in ("base_url", "api_key", "query", "content", "prompt", "raw_response"):
            self.assertNotIn(f'"{forbidden}"', serialized)

    def test_prediction_contract_rejects_extra_nonfinite_rank_and_type_confusion(self):
        prepared, predictions = self._prediction_fixture()
        mutations = []
        for path, value in (
            (("configuration", "extra"), True),
            (("profile", "extra"), True),
            (("index", "extra"), True),
            (("tasks", 0, "diagnostics", "extra"), True),
            (("tasks", 0, "rankings", 0, 0, "extra"), True),
            (("tasks", 0, "latency_ms", 0), float("nan")),
            (("tasks", 0, "rankings", 0, 0, "rrf_score"), float("inf")),
            (("tasks", 0, "rankings", 0, 0, "keyword_rank"), 0),
            (("tasks", 0, "rankings", 0, 0, "line_start"), "1"),
        ):
            candidate = copy.deepcopy(predictions)
            target = candidate
            for segment in path[:-1]:
                target = target[segment]
            target[path[-1]] = value
            mutations.append(candidate)
        for candidate in mutations:
            with self.subTest():
                with self.assertRaises(RetrievalRunnerError):
                    validate_predictions(candidate, prepared)

    def test_atomic_writer_preserves_old_file_on_replace_failure_and_rejects_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            path.write_text("old", encoding="utf-8")
            with patch(
                "scripts.run_evidence_retrieval_eval.os.replace",
                side_effect=OSError("fixture"),
            ):
                with self.assertRaises(OSError):
                    _write_json(path, {"new": True})
            self.assertEqual("old", path.read_text(encoding="utf-8"))
            self.assertEqual([], list(path.parent.glob(".*.tmp")))
            with self.assertRaises(RetrievalRunnerError):
                _assert_distinct_paths(path, path)
            alias = Path(temporary) / "hardlink.json"
            os.link(path, alias)
            with self.assertRaises(RetrievalRunnerError):
                _assert_distinct_paths(path, alias)

    def test_prepare_pins_both_manifest_and_freeze_hashes(self):
        self.assertEqual(
            PINNED_MANIFEST_SHA256,
            hashlib.sha256((DEFAULT_SUITE_ROOT / "manifest.json").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            PINNED_FREEZE_SHA256,
            hashlib.sha256((DEFAULT_SUITE_ROOT / "freeze.json").read_bytes()).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "suite"
            shutil.copytree(DEFAULT_SUITE_ROOT, copied)
            freeze_path = copied / "freeze.json"
            freeze_path.write_bytes(freeze_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(RetrievalRunnerError, "pinned v1 hashes"):
                prepare_suite(suite_root=copied, split="dev")

    def test_tracked_dirty_gate_ignores_untracked_but_rejects_tracked_changes(self):
        with patch(
            "scripts.run_evidence_retrieval_eval.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout=""),
        ):
            _assert_clean_tracked_worktree()
        with patch(
            "scripts.run_evidence_retrieval_eval.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout=" M app/example.py\n"),
        ):
            with self.assertRaisesRegex(RetrievalRunnerError, "must be clean"):
                _assert_clean_tracked_worktree()

    def test_compare_requires_common_provenance_and_emits_conservative_gate(self):
        prepared, predictions = self._prediction_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path = root / "prepared.json"
            predictions_path = root / "predictions.json"
            prepared_sha = _write_json(prepared_path, prepared)
            predictions["prepared_sha256"] = prepared_sha
            prediction_sha = _write_json(predictions_path, predictions)
            baseline = score_prediction_files(
                prepared_path=prepared_path,
                predictions_path=predictions_path,
                expected_prediction_sha256=prediction_sha,
                manifest_path=DEFAULT_SUITE_ROOT / "manifest.json",
            )
            reports = []
            hashes = []
            for strategy, misses in (
                ("keyword-only", 3),
                ("dense-only", 2),
                ("keyword+dense-rrf", 0),
            ):
                report = copy.deepcopy(baseline)
                report["configuration"]["strategy"] = strategy
                report["index"]["provider"]["used_for_index"] = strategy != "keyword-only"
                for index, task in enumerate(report["tasks"]):
                    hit = index >= misses
                    task["recalled"] = task["expected"] if hit else 0
                    task["all_evidence"] = hit
                    task["reciprocal_rank"] = 1.0 if hit else 0.0
                    task["status"] = "completed"
                    task["fallback"] = False
                    task["unstable"] = False
                    task["isolation_failures"] = 0
                    task["latency_ms"] = [1.0, 1.0, 1.0]
                aggregates = _aggregate_task_scores(report["tasks"])
                report["valid_run"] = aggregates["valid_run"]
                report["metrics"] = aggregates["metrics"]
                report["per_label"] = aggregates["per_label"]
                path = root / f"{strategy}.json"
                hashes.append(_write_json(path, report))
                reports.append(path)
            comparison = compare_score_reports(reports, hashes)
            self.assertTrue(comparison["gate"]["passed"])
            self.assertTrue(comparison["hybrid_improvement_claim"]["supported"])
            mismatched = json.loads(reports[1].read_text(encoding="utf-8"))
            mismatched["code_commit"] = "c" * 40
            hashes[1] = _write_json(reports[1], mismatched)
            with self.assertRaisesRegex(RetrievalRunnerError, "provenance"):
                compare_score_reports(reports, hashes)

    def test_trimmed_prepared_and_hand_edited_aggregate_are_rejected(self):
        trimmed = copy.deepcopy(self.prepared)
        trimmed["tasks"].pop()
        with self.assertRaisesRegex(RetrievalRunnerError, "complete pinned split"):
            run_prepared(
                trimmed,
                prepared_sha256=hashlib.sha256(_canonical_bytes(trimmed)).hexdigest(),
                configuration=self.configuration,
                executor=_FakeExecutor(trimmed, self.configuration, self.profile),
                canonical_profile=self.profile,
                code_commit="d" * 40,
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trimmed_path = root / "trimmed.json"
            invalid_predictions = root / "invalid-predictions.json"
            _write_json(trimmed_path, trimmed)
            invalid_sha = _write_json(invalid_predictions, {})
            with self.assertRaisesRegex(RetrievalRunnerError, "complete pinned split"):
                score_prediction_files(
                    prepared_path=trimmed_path,
                    predictions_path=invalid_predictions,
                    expected_prediction_sha256=invalid_sha,
                    manifest_path=DEFAULT_SUITE_ROOT / "manifest.json",
                )
        prepared, predictions = self._prediction_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path = root / "prepared.json"
            predictions_path = root / "predictions.json"
            prepared_sha = _write_json(prepared_path, prepared)
            predictions["prepared_sha256"] = prepared_sha
            prediction_sha = _write_json(predictions_path, predictions)
            report = score_prediction_files(
                prepared_path=prepared_path,
                predictions_path=predictions_path,
                expected_prediction_sha256=prediction_sha,
                manifest_path=DEFAULT_SUITE_ROOT / "manifest.json",
            )
        report["metrics"]["evidence_recall_at_5"] = 1.0
        with self.assertRaisesRegex(RetrievalRunnerError, "metrics are inconsistent"):
            validate_score_report(report)


if __name__ == "__main__":
    unittest.main()
