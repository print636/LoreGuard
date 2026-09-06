from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import run_issue_review_eval as runner


def _bytes(value: dict) -> bytes:
    return runner._canonical_bytes(value)


def _write(path: Path, value: dict) -> str:
    payload = _bytes(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _configuration(mode: str, repeats: int = 1) -> runner.RuntimeConfiguration:
    return runner.RuntimeConfiguration(
        mode=mode,
        repeats=repeats,
        chat_model="mock-chat-v1",
        chat_endpoint_fingerprint="endpoint-sha256:" + "a" * 64,
        thinking_mode="disabled",
        top_k=6,
        token_budget=6_000,
        timeout_seconds=20,
        total_deadline_seconds=45,
    )


class _OracleFixtureExecutor:
    def __init__(self, package: dict, *, manifest: dict, fail: bool = False, unstable: bool = False):
        self.metadata = {
            "backend": "mock-contract",
            "parser_contract": "issue-evidence-review-v1",
            "citation_validation": "production-strict",
            "embedding": {"kind": "mock"},
        }
        self._package = {case["case_id"]: case for case in package["cases"]}
        self._manifest_cases = {case["case_id"]: case for case in manifest["cases"]}
        self._evidence = {row["evidence_id"]: row for row in manifest["evidence"]}
        self._fail = fail
        self._unstable = unstable
        self._calls: dict[str, int] = {}

    def execute(self, case: dict, mode: str) -> dict:
        count = self._calls.get(case["case_id"], 0) + 1
        self._calls[case["case_id"]] = count
        if self._fail:
            return {
                "status": "degraded",
                "verdict": None,
                "citations": [],
                "review_outcome": "degraded",
                "failure_category": "hybrid_unavailable",
                "retrieval_mode": "lexical_only",
                "retrieval_strategy": "keyword+dense-rrf",
                "latency_ms": 7,
                "prompt_tokens": 9,
                "completion_tokens": 0,
                "charged_tokens": 9,
                "provider_calls": [],
            }
        oracle = self._manifest_cases[case["case_id"]]
        expected = oracle["expected"]
        verdict = expected["verdict"]
        if self._unstable and count % 2 == 0:
            verdict = "supports_issue" if verdict != "supports_issue" else "insufficient_evidence"
        documents = {
            (row["document_id"], row["document_version"]): row
            for row in case["documents"]
        }
        citations = []
        for evidence_id in expected["citation_evidence_ids"]:
            evidence = self._evidence[evidence_id]
            document = documents[(evidence["document_id"], evidence["version"])]
            citations.append(
                {
                    "project_id": case["project_id"],
                    "document_id": evidence["document_id"],
                    "document_version": evidence["version"],
                    "content_sha256": document["content_sha256"],
                    "line_start": evidence["start_line"],
                    "line_end": evidence["end_line"],
                }
            )
        return {
            "status": "success",
            "verdict": verdict,
            "citations": citations,
            "review_outcome": "completed",
            "failure_category": None,
            "retrieval_mode": "local_only" if mode == "local-context" else "hybrid",
            "retrieval_strategy": "local-context-only" if mode == "local-context" else "keyword+dense-rrf",
            "latency_ms": 11,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "charged_tokens": 120,
            "provider_calls": [{"status": "success", "purpose": "evidence_review"}],
        }


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(runner.DEFAULT_MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture()
def dev_package() -> dict:
    return runner.prepare_execution_package(split="dev")


def _contains_forbidden_key(value) -> bool:
    if isinstance(value, dict):
        return any(
            key in runner._FORBIDDEN_EXECUTION_KEYS or _contains_forbidden_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def test_prepare_rebuilds_complete_dev_without_oracle_fields(dev_package):
    assert dev_package["schema_version"] == runner.EXECUTION_SCHEMA
    assert dev_package["split"] == "dev"
    assert len(dev_package["cases"]) == 6
    assert not _contains_forbidden_key(dev_package)
    assert all(case["current_evidence"] for case in dev_package["cases"])
    assert all(case["allowed_snapshots"] for case in dev_package["cases"])
    runner.validate_execution_package(dev_package)


def test_prepare_full_contract_covers_both_frozen_splits_without_scoring():
    package = runner.prepare_execution_package(split="full")
    assert package["split"] == "full"
    assert len(package["cases"]) == 12
    assert {case["split"] for case in package["cases"]} == {"dev", "holdout"}
    assert not _contains_forbidden_key(package)
    runner.validate_execution_package(package)


def test_all_prepared_canonical_execution_hashes_are_pinned():
    for split in ("dev", "holdout", "full"):
        package = runner.prepare_execution_package(split=split)
        assert hashlib.sha256(_bytes(package)).hexdigest() == runner.PINNED_EXECUTION_SHA256[split]


def test_prepare_fails_if_any_frozen_byte_changes(tmp_path, manifest):
    manifest_path = tmp_path / "manifest.json"
    freeze_path = tmp_path / "freeze.json"
    manifest_path.write_bytes(runner.DEFAULT_MANIFEST.read_bytes() + b" ")
    freeze_path.write_bytes(runner.DEFAULT_FREEZE.read_bytes())
    with pytest.raises(runner.EvalContractError, match="pinned"):
        runner.prepare_execution_package(
            manifest_path=manifest_path,
            freeze_path=freeze_path,
            split="dev",
        )


def test_prepare_rejects_any_freeze_byte_change(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    freeze_path = tmp_path / "freeze.json"
    manifest_path.write_bytes(runner.DEFAULT_MANIFEST.read_bytes())
    freeze_path.write_bytes(runner.DEFAULT_FREEZE.read_bytes() + b" ")
    with pytest.raises(runner.EvalContractError, match="freeze does not match the pinned"):
        runner.prepare_execution_package(
            manifest_path=manifest_path,
            freeze_path=freeze_path,
            split="dev",
        )


def test_trimmed_or_modified_execution_rehash_still_fails_pinned_contract(dev_package, manifest):
    variants = []
    trimmed = copy.deepcopy(dev_package)
    trimmed["cases"].pop()
    variants.append(trimmed)
    modified = copy.deepcopy(dev_package)
    modified["cases"][0]["candidate_issue"]["summary"] += "篡改"
    variants.append(modified)
    for forged in variants:
        with pytest.raises(runner.EvalContractError, match="pinned suite"):
            runner.run_execution_package(
                forged,
                execution_sha256=hashlib.sha256(_bytes(forged)).hexdigest(),
                configuration=_configuration("local-context"),
                executor=_OracleFixtureExecutor(forged, manifest=manifest),
                code_commit="8" * 40,
            )


def test_run_rejects_extra_oracle_field_before_executor(dev_package):
    forged = copy.deepcopy(dev_package)
    forged["cases"][0]["expected"] = {"verdict": "supports_issue"}

    class Forbidden:
        metadata = {}

        def execute(self, case, mode):
            raise AssertionError("executor must not run")

    with pytest.raises(runner.EvalContractError, match="oracle"):
        runner.run_execution_package(
            forged,
            execution_sha256=hashlib.sha256(_bytes(forged)).hexdigest(),
            configuration=_configuration("local-context"),
            executor=Forbidden(),
            code_commit="a" * 40,
        )


def test_run_records_configuration_and_three_stable_repeats(dev_package, manifest):
    executor = _OracleFixtureExecutor(dev_package, manifest=manifest)
    digest = hashlib.sha256(_bytes(dev_package)).hexdigest()
    predictions = runner.run_execution_package(
        dev_package,
        execution_sha256=digest,
        configuration=_configuration("rag-evidence", repeats=3),
        executor=executor,
        code_commit="b" * 40,
    )
    assert predictions["configuration"] == _configuration("rag-evidence", 3).safe_dict()
    assert predictions["executor_metadata"]["parser_contract"] == "issue-evidence-review-v1"
    assert all(len(case["runs"]) == 3 for case in predictions["cases"])
    runner.validate_predictions(predictions, dev_package)


def test_score_hashes_predictions_before_loading_oracle(tmp_path, dev_package):
    execution_path = tmp_path / "execution.json"
    prediction_path = tmp_path / "predictions.json"
    execution_sha = _write(execution_path, dev_package)
    _write(prediction_path, {"schema_version": runner.PROVIDER_CHECK_SCHEMA})
    with pytest.raises(runner.EvalContractError, match="prediction artifact hash"):
        runner.score_predictions(
            prediction_path=prediction_path,
            prediction_sha256="0" * 64,
            execution_path=execution_path,
            execution_sha256=execution_sha,
            manifest_path=tmp_path / "oracle-must-not-be-read.json",
            freeze_path=tmp_path / "freeze-must-not-be-read.json",
        )


def test_provider_check_artifact_cannot_masquerade_as_predictions(tmp_path, dev_package):
    execution_path = tmp_path / "execution.json"
    prediction_path = tmp_path / "provider-check.json"
    execution_sha = _write(execution_path, dev_package)
    prediction_sha = _write(
        prediction_path,
        {"schema_version": runner.PROVIDER_CHECK_SCHEMA, "evaluation_eligible": False},
    )
    with pytest.raises(runner.EvalContractError, match="predictions schema"):
        runner.score_predictions(
            prediction_path=prediction_path,
            prediction_sha256=prediction_sha,
            execution_path=execution_path,
            execution_sha256=execution_sha,
        )


def test_score_rejects_rehashed_modified_execution_before_oracle(tmp_path, dev_package, manifest):
    original_sha = hashlib.sha256(_bytes(dev_package)).hexdigest()
    predictions = runner.run_execution_package(
        dev_package,
        execution_sha256=original_sha,
        configuration=_configuration("local-context"),
        executor=_OracleFixtureExecutor(dev_package, manifest=manifest),
        code_commit="7" * 40,
    )
    prediction_path = tmp_path / "predictions.json"
    prediction_sha = _write(prediction_path, predictions)
    forged = copy.deepcopy(dev_package)
    forged["cases"][0]["candidate_issue"]["summary"] += "篡改"
    execution_path = tmp_path / "forged-execution.json"
    forged_sha = _write(execution_path, forged)
    with pytest.raises(runner.EvalContractError, match="pinned suite"):
        runner.score_predictions(
            prediction_path=prediction_path,
            prediction_sha256=prediction_sha,
            execution_path=execution_path,
            execution_sha256=forged_sha,
            manifest_path=tmp_path / "must-not-load-manifest.json",
            freeze_path=tmp_path / "must-not-load-freeze.json",
        )


def test_score_reports_verdict_citations_classes_latency_and_tokens(tmp_path, dev_package, manifest):
    execution_path = tmp_path / "execution.json"
    prediction_path = tmp_path / "predictions.json"
    execution_sha = _write(execution_path, dev_package)
    predictions = runner.run_execution_package(
        dev_package,
        execution_sha256=execution_sha,
        configuration=_configuration("rag-evidence", repeats=3),
        executor=_OracleFixtureExecutor(dev_package, manifest=manifest),
        code_commit="c" * 40,
    )
    prediction_sha = _write(prediction_path, predictions)
    report = runner.score_predictions(
        prediction_path=prediction_path,
        prediction_sha256=prediction_sha,
        execution_path=execution_path,
        execution_sha256=execution_sha,
    )
    summary = report["summary"]
    assert summary["correct_cases_all_repeats"] == 6
    assert summary["minimal_citation_coverage_cases_all_repeats"] == 6
    assert summary["stable_cases"] == 6
    assert summary["non_allowlisted_runs"] == 0
    assert summary["excluded_evidence_leaks"] == 0
    assert summary["failed_runs"] == 0
    assert summary["degraded_runs"] == 0
    assert summary["latency_ms"] == {"count": 18, "mean": 11, "p95": 11}
    assert summary["prompt_tokens"] == 1_800
    assert summary["by_expected_class"]["contextual_exception"] == {
        "cases": 2,
        "correct_cases": 2,
    }
    assert report["absolute_gate"] == {
        "evaluable": False,
        "passed": None,
        "policy": "frozen-readme-v1",
        "reason": "full_12_case_suite_required",
    }
    assert "content" not in json.dumps(report, ensure_ascii=False)


def test_degraded_and_unstable_runs_never_count_as_correct(tmp_path, dev_package, manifest):
    execution_path = tmp_path / "execution.json"
    prediction_path = tmp_path / "predictions.json"
    execution_sha = _write(execution_path, dev_package)
    degraded = runner.run_execution_package(
        dev_package,
        execution_sha256=execution_sha,
        configuration=_configuration("rag-evidence", repeats=1),
        executor=_OracleFixtureExecutor(dev_package, manifest=manifest, fail=True),
        code_commit="d" * 40,
    )
    prediction_sha = _write(prediction_path, degraded)
    report = runner.score_predictions(
        prediction_path=prediction_path,
        prediction_sha256=prediction_sha,
        execution_path=execution_path,
        execution_sha256=execution_sha,
    )
    assert report["summary"]["correct_cases_all_repeats"] == 0
    assert report["summary"]["degraded_runs"] == 6

    unstable = runner.run_execution_package(
        dev_package,
        execution_sha256=execution_sha,
        configuration=_configuration("rag-evidence", repeats=3),
        executor=_OracleFixtureExecutor(dev_package, manifest=manifest, unstable=True),
        code_commit="d" * 40,
    )
    unstable_sha = _write(prediction_path, unstable)
    unstable_report = runner.score_predictions(
        prediction_path=prediction_path,
        prediction_sha256=unstable_sha,
        execution_path=execution_path,
        execution_sha256=execution_sha,
    )
    assert unstable_report["summary"]["correct_cases_all_repeats"] == 0
    assert unstable_report["summary"]["stable_cases"] == 0


def test_forged_snapshot_hash_fails_allowlist_and_gold_coverage(tmp_path, dev_package, manifest):
    execution_path = tmp_path / "execution.json"
    prediction_path = tmp_path / "predictions.json"
    execution_sha = _write(execution_path, dev_package)
    predictions = runner.run_execution_package(
        dev_package,
        execution_sha256=execution_sha,
        configuration=_configuration("rag-evidence"),
        executor=_OracleFixtureExecutor(dev_package, manifest=manifest),
        code_commit="9" * 40,
    )
    predictions["cases"][0]["runs"][0]["citations"][0]["content_sha256"] = "0" * 64
    prediction_sha = _write(prediction_path, predictions)
    report = runner.score_predictions(
        prediction_path=prediction_path,
        prediction_sha256=prediction_sha,
        execution_path=execution_path,
        execution_sha256=execution_sha,
    )
    first = report["cases"][0]
    assert first["citation_allowlist_ok_all_repeats"] is False
    assert first["minimal_citation_coverage_all_repeats"] is False
    assert report["summary"]["citation_allowlist"]["allowed"] < report["summary"]["citation_allowlist"]["total"]


def test_strict_prediction_schema_turns_executor_extras_into_failure(dev_package):
    class ExtraFieldExecutor:
        metadata = {"backend": "mock"}

        def execute(self, case, mode):
            return {"raw_response": "must not escape"}

    predictions = runner.run_execution_package(
        dev_package,
        execution_sha256=hashlib.sha256(_bytes(dev_package)).hexdigest(),
        configuration=_configuration("local-context"),
        executor=ExtraFieldExecutor(),
        code_commit="e" * 40,
    )
    assert all(case["runs"][0]["status"] == "failed" for case in predictions["cases"])
    assert "must not escape" not in repr(predictions)


def test_compare_requires_hashes_and_identical_chat_contract(tmp_path, dev_package, manifest):
    execution_path = tmp_path / "execution.json"
    execution_sha = _write(execution_path, dev_package)
    reports = {}
    for mode in ("local-context", "rag-evidence"):
        predictions = runner.run_execution_package(
            dev_package,
            execution_sha256=execution_sha,
            configuration=_configuration(mode),
            executor=_OracleFixtureExecutor(dev_package, manifest=manifest),
            code_commit="f" * 40,
        )
        prediction_path = tmp_path / f"{mode}.predictions.json"
        prediction_sha = _write(prediction_path, predictions)
        report = runner.score_predictions(
            prediction_path=prediction_path,
            prediction_sha256=prediction_sha,
            execution_path=execution_path,
            execution_sha256=execution_sha,
        )
        report_path = tmp_path / f"{mode}.report.json"
        report_sha = _write(report_path, report)
        reports[mode] = (report_path, report_sha)
    comparison = runner.compare_reports(
        local_path=reports["local-context"][0],
        local_sha256=reports["local-context"][1],
        rag_path=reports["rag-evidence"][0],
        rag_sha256=reports["rag-evidence"][1],
    )
    assert comparison["gains"] == {
        "verdict_correct_cases": 0,
        "contextual_exception_correct_cases": 0,
        "minimal_citation_coverage_cases": 0,
    }
    assert comparison["gate"]["evaluable"] is False
    with pytest.raises(runner.EvalContractError, match="report artifact hash"):
        runner.compare_reports(
            local_path=reports["local-context"][0],
            local_sha256="0" * 64,
            rag_path=reports["rag-evidence"][0],
            rag_sha256=reports["rag-evidence"][1],
        )

    tampered = json.loads(reports["local-context"][0].read_text(encoding="utf-8"))
    tampered["summary"]["correct_cases_all_repeats"] -= 1
    tampered_path = tmp_path / "tampered-summary.report.json"
    tampered_sha = _write(tampered_path, tampered)
    with pytest.raises(runner.EvalContractError, match="summary was not derived"):
        runner.compare_reports(
            local_path=tampered_path,
            local_sha256=tampered_sha,
            rag_path=reports["rag-evidence"][0],
            rag_sha256=reports["rag-evidence"][1],
        )

    tampered_gate = json.loads(reports["local-context"][0].read_text(encoding="utf-8"))
    tampered_gate["absolute_gate"]["evaluable"] = True
    gate_path = tmp_path / "tampered-gate.report.json"
    gate_sha = _write(gate_path, tampered_gate)
    with pytest.raises(runner.EvalContractError, match="gate was not derived"):
        runner.compare_reports(
            local_path=gate_path,
            local_sha256=gate_sha,
            rag_path=reports["rag-evidence"][0],
            rag_sha256=reports["rag-evidence"][1],
        )


def test_atomic_writer_cannot_touch_frozen_data_or_inputs(tmp_path, dev_package):
    source = tmp_path / "source.json"
    _write(source, dev_package)
    with pytest.raises(runner.EvalContractError, match="overwrite an input"):
        runner._atomic_write_json(source, dev_package, protected=(source,))
    with pytest.raises(runner.EvalContractError, match="frozen dataset"):
        runner._atomic_write_json(
            runner.DATASET_ROOT / "forbidden-output.json",
            dev_package,
            protected=(),
        )


def test_real_cli_requires_execute_before_reading_files():
    with pytest.raises(runner.EvalContractError, match="explicit --execute"):
        runner.main(
            [
                "run",
                "--execution",
                "missing.json",
                "--execution-sha256",
                "0" * 64,
                "--output",
                "never.json",
                "--mode",
                "local-context",
                "--chat-base-url",
                "https://example.invalid/v1",
                "--chat-api-key-file",
                "missing.key",
                "--chat-model",
                "mock",
            ]
        )


def test_secret_loader_rejects_dotenv(tmp_path):
    path = tmp_path / ".env"
    path.write_text("do-not-read", encoding="utf-8")
    with pytest.raises(runner.EvalContractError, match=".env files are forbidden"):
        runner._read_secret_file(path)


def test_embedding_insecure_http_requires_explicit_cli_opt_in_and_is_reported():
    base = [
        "run", "--execution", "execution.json", "--execution-sha256", "0" * 64,
        "--output", "predictions.json", "--mode", "rag-evidence",
        "--chat-base-url", "https://example.invalid/v1",
        "--chat-api-key-file", "chat.key", "--chat-model", "mock",
    ]
    disabled = runner._parser().parse_args(base)
    enabled = runner._parser().parse_args(base + ["--embedding-allow-insecure-http"])
    assert disabled.embedding_allow_insecure_http is False
    assert enabled.embedding_allow_insecure_http is True
    assert _configuration("rag-evidence").safe_dict()["embedding_allow_insecure_http"] is False
    explicitly_enabled = replace(
        _configuration("rag-evidence"), embedding_allow_insecure_http=True
    )
    assert explicitly_enabled.safe_dict()["embedding_allow_insecure_http"] is True


def test_compare_allows_local_https_default_and_rag_explicit_http_opt_in(
    tmp_path, dev_package, manifest
):
    execution_path = tmp_path / "execution.json"
    execution_sha = _write(execution_path, dev_package)
    reports = {}
    for mode, allow_http in (("local-context", False), ("rag-evidence", True)):
        configuration = replace(
            _configuration(mode), embedding_allow_insecure_http=allow_http
        )
        predictions = runner.run_execution_package(
            dev_package,
            execution_sha256=execution_sha,
            configuration=configuration,
            executor=_OracleFixtureExecutor(dev_package, manifest=manifest),
            code_commit="6" * 40,
        )
        prediction_path = tmp_path / f"{mode}.predictions.json"
        prediction_sha = _write(prediction_path, predictions)
        report = runner.score_predictions(
            prediction_path=prediction_path,
            prediction_sha256=prediction_sha,
            execution_path=execution_path,
            execution_sha256=execution_sha,
        )
        report_path = tmp_path / f"{mode}.report.json"
        reports[mode] = (report_path, _write(report_path, report))

    comparison = runner.compare_reports(
        local_path=reports["local-context"][0],
        local_sha256=reports["local-context"][1],
        rag_path=reports["rag-evidence"][0],
        rag_sha256=reports["rag-evidence"][1],
    )
    assert comparison["schema_version"] == runner.COMPARISON_SCHEMA
