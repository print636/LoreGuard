from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import UUID

import httpx
import pytest

from scripts import ooc_eval_contract as contract
from scripts import run_ooc_sealed_http as runner


def _uuid(number: int) -> str:
    return str(UUID(f"00000000-0000-4000-8000-{number:012d}"))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_bundle(root: Path):
    root.mkdir()
    dataset_id, group_id, world_id = _uuid(1), _uuid(2), _uuid(3)
    case_id, axis_id = _uuid(4), _uuid(5)
    baseline_id, target_id = _uuid(6), _uuid(7)
    baseline = "林澈长期保持克制，不会公开发怒。\n"
    target = (
        "林澈当众掀翻会议桌。\n"
        "次日，林澈再次怒斥无关人员。\n"
        "第三日，林澈在走廊砸碎了花瓶。\n"
    )
    files = {baseline_id: baseline, target_id: target}
    snapshots = []
    evidence = []
    evidence_number = 20
    for document_id, content in files.items():
        relative = f"cases/{group_id}/{document_id}.md"
        path = root / Path(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        snapshots.append(
            contract.DocumentSnapshot(
                document_id=document_id,
                path=relative,
                version=1,
                sha256=_sha(content),
            )
        )
        for line_number, line in enumerate(content.splitlines(), 1):
            if line.strip():
                evidence.append(
                    contract.EvidenceAnchor(
                        evidence_id=_uuid(evidence_number),
                        document_id=document_id,
                        line_start=line_number,
                        line_end=line_number,
                    )
                )
                evidence_number += 1
    selector = runner.PublicCandidateSelector(
        character_key="林澈",
        trait_type="core_personality",
        source_document_id=baseline_id,
        source_line_start=1,
        source_line_end=1,
        origin="explicit_setting",
        polarity="positive",
        stability="stable",
    )
    author_axis = runner.AuthorAxis(
        display_name="情绪克制",
        definition="角色在公开冲突中保持克制",
        positive_proposition="角色面对冲突时保持克制",
        applicability_scope="公开冲突情境",
        axis_alignment="same",
    )
    axis_payload = {
        "candidate_selector": selector.model_dump(mode="json"),
        "author_axis": author_axis.model_dump(mode="json"),
    }
    public_case = contract.PublicInputCase(
        case_id=case_id,
        split_id="closed-pilot-v1",
        group_id=group_id,
        world_id=world_id,
        documents=tuple(snapshots),
        axis=contract.AxisSnapshot(
            axis_id=axis_id,
            version=1,
            sha256=runner._canonical_value_sha(axis_payload),
        ),
        evidence_catalog=tuple(evidence),
    )
    public = contract.PublicInputBundle(dataset_id=dataset_id, cases=(public_case,))
    setup_case = runner.PublicAuthorSetupCase(
        case_id=case_id,
        group_id=group_id,
        target_document_id=target_id,
        documents=(
            runner.PublicDocumentSetup(
                document_id=baseline_id,
                phase="baseline",
                document_role="character_profile",
                story_scope="main",
                resolution_state="confirmed",
                publication_status="published",
                import_order=1,
            ),
            runner.PublicDocumentSetup(
                document_id=target_id,
                phase="target",
                document_role="chapter",
                story_scope="main",
                resolution_state="draft",
                publication_status="draft",
                import_order=2,
            ),
        ),
        candidate_selector=selector,
        author_axis=author_axis,
    )
    setup = runner.PublicAuthorSetupBundle(
        dataset_id=dataset_id,
        public_input_sha256=contract.canonical_sha256(public),
        cases=(setup_case,),
    )
    freeze = runner.ExecutionFreeze(
        dataset_id=dataset_id,
        public_input_sha256=contract.canonical_sha256(public),
        author_setup_sha256=runner._canonical_model_sha(setup),
        case_count=1,
        group_count=1,
    )
    for name, value in (
        ("public-input.json", public),
        ("author-setup.json", setup),
        ("freeze.json", freeze),
    ):
        (root / name).write_text(
            json.dumps(value.model_dump(mode="json"), ensure_ascii=False),
            encoding="utf-8",
        )
    return runner.load_public_bundle(root), public_case, setup_case


def _config() -> runner.SealedHttpRunConfig:
    runtime = {
        "capabilities": {"character_consistency": True},
        "character_consistency_limits": {
            "signal_support_id_v4": True,
            "signal_full_line_echo_v2": True,
        },
    }
    return runner.SealedHttpRunConfig(
        base_url="http://127.0.0.1:8101",
        isolation=runner.IsolationExpectation(
            eval_db_id=_uuid(90),
            database_path_sha256="a" * 64,
            instance_id_sha256="b" * 64,
        ),
        execution_id=_uuid(91),
        repeat_id="repeat-1",
        expected_runtime_provenance_sha256=runner._canonical_value_sha(runtime),
        expected_runner_source_sha256=runner.current_runner_source_sha256(),
        run_timeout_seconds=30.0,
        poll_interval_seconds=0.01,
    )


class SuccessfulApi:
    def __init__(
        self,
        bundle: runner.LoadedBundle,
        *,
        duplicate=False,
        extra_unmatched=False,
        report_evidence_mode: str | None = None,
    ):
        self.bundle = bundle
        self.duplicate = duplicate
        self.extra_unmatched = extra_unmatched
        self.report_evidence_mode = report_evidence_mode
        self.decisions: list[tuple[str, str]] = []
        self.project_id = _uuid(100)
        self.baseline_run_id = _uuid(101)
        self.draft_run_id = _uuid(102)
        self.baseline_runtime_id = _uuid(103)
        self.target_runtime_id = _uuid(104)
        self.candidate_id = _uuid(105)
        self.axis_id = _uuid(106)
        self.decision = 120
        self.runtime = {
            "capabilities": {"character_consistency": True},
            "character_consistency_limits": {
                "signal_support_id_v4": True,
                "signal_full_line_echo_v2": True,
            },
        }

    def _json(self, value, status=200):
        return httpx.Response(status, json=value)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        case = self.bundle.public.cases[0]
        setup = self.bundle.setup.cases[0]
        baseline_public = next(
            row for row in case.documents
            if row.document_id == setup.candidate_selector.source_document_id
        )
        target_public = next(
            row for row in case.documents if row.document_id == setup.target_document_id
        )
        if path == "/api/v1/evaluation/isolation-identity":
            return self._json({
                **_config().isolation.model_dump(mode="json"),
                "fresh_for_prepare": not hasattr(self, "created"),
            })
        if path == "/health":
            return self._json({
                "model": {"configured": True},
                "runtime_provenance": self.runtime,
            })
        if method == "POST" and path == "/api/v1/projects":
            self.created = True
            return self._json({"id": self.project_id}, 201)
        if method == "POST" and path.endswith("/documents/text"):
            body = json.loads(request.content)
            is_target = body["name"] == f"{target_public.document_id}.md"
            runtime_id = self.target_runtime_id if is_target else self.baseline_runtime_id
            return self._json({
                "id": runtime_id,
                "project_id": self.project_id,
                "name": body["name"],
                "version": 1,
                "content": body["content"],
                "document_role": body["document_role"],
                "story_scope": body["story_scope"],
            }, 201)
        if method == "POST" and path.endswith("/analysis-runs"):
            body = json.loads(request.content)
            return self._json({
                "id": self.baseline_run_id if body["mode"] == "baseline_build"
                else self.draft_run_id,
            }, 202)
        if method == "GET" and path in {
            f"/api/v1/analysis-runs/{self.baseline_run_id}",
            f"/api/v1/analysis-runs/{self.draft_run_id}",
        }:
            return self._json({
                "id": path.rsplit("/", 1)[1],
                "status": "completed",
            })
        if method == "GET" and path.endswith("/diagnostics"):
            is_draft = self.draft_run_id in path
            stage = {
                "outcome": "completed",
                "material_coverage": "complete",
                "counts": {
                    "planned_chunks": 1,
                    "processed_chunks": 1,
                    "model_incomplete_chunks": 0,
                    "model_uncalled_chunks": 0,
                },
                "case_trace": [],
            }
            if is_draft:
                citation_refs = [
                    {"handle": "B01", "role": "B",
                     "document_name": f"{baseline_public.document_id}.md",
                     "line_start": 1, "line_end": 1},
                    {"handle": "C01", "role": "C",
                     "document_name": f"{target_public.document_id}.md",
                     "line_start": 1, "line_end": 1},
                    {"handle": "C02", "role": "C",
                     "document_name": f"{target_public.document_id}.md",
                     "line_start": 2, "line_end": 2},
                ]
                independent = ["C01", "C02"]
                if self.report_evidence_mode == "trace_extra_c":
                    citation_refs.append({
                        "handle": "C03", "role": "C",
                        "document_name": f"{target_public.document_id}.md",
                        "line_start": 3, "line_end": 3,
                    })
                    independent.append("C03")
                stage["case_trace"] = [{
                    "confirmed_candidate_id_sha256": _sha(self.candidate_id),
                    "material_coverage": "complete",
                    "explanation_coverage": "complete",
                    "final_outcome": "conflict",
                    "citation_refs_incomplete": False,
                    "event_independence": "yes",
                    "independent_event_citations": independent,
                    "citation_refs": citation_refs,
                }]
            return self._json({
                "runtime_provenance": self.runtime,
                "character_consistency": stage,
            })
        if method == "GET" and path.endswith("/characters"):
            return self._json({
                "items": [{"character_key": setup.candidate_selector.character_key}],
                "has_more": False,
            })
        if method == "GET" and path.endswith("/profile-candidates"):
            candidates = [self._candidate(baseline_public, setup)]
            if self.duplicate:
                candidates.append({**candidates[0], "id": _uuid(107)})
            if self.extra_unmatched:
                candidates.append({
                    **candidates[0], "id": _uuid(108), "trait_type": "preference"
                })
            return self._json({"items": candidates, "has_more": False})
        if method == "POST" and path.endswith("/character-trait-axes"):
            body = json.loads(request.content)
            return self._json({
                "id": self.axis_id,
                "project_id": self.project_id,
                "version": 1,
                "trait_type": body["trait_type"],
                "definition_sha256": _sha(body["definition"]),
                "positive_proposition_sha256": _sha(body["positive_proposition"]),
                "comparison_key": None,
                "applicability_scope_sha256": None,
            }, 201)
        if method == "POST" and path.endswith("/decisions"):
            body = json.loads(request.content)
            candidate_id = path.split("/")[-2]
            self.decisions.append((candidate_id, body["decision"]))
            state = "confirmed" if body["decision"] == "confirm" else "rejected"
            return self._json({
                "candidate": {
                    "id": candidate_id,
                    "review_state": state,
                    "revision": body["expected_revision"] + 1,
                },
                "decision_id": _uuid(self.decision),
                "deduplicated": False,
            }, 201)
        if method == "GET" and path.endswith("/issues"):
            evidence = [
                {"document_name": f"{baseline_public.document_id}.md",
                 "line_start": 1, "line_end": 1},
                {"document_name": f"{target_public.document_id}.md",
                 "line_start": 1, "line_end": 1},
                {"document_name": f"{target_public.document_id}.md",
                 "line_start": 2, "line_end": 2},
            ]
            review_citation_refs = [
                {"handle": "B01", "role": "B", "evidence_index": 0,
                 "response_index": 0},
                {"handle": "C01", "role": "C", "evidence_index": 1,
                 "response_index": 1},
                {"handle": "C02", "role": "C", "evidence_index": 2,
                 "response_index": 2},
            ]
            if self.report_evidence_mode == "empty":
                evidence = []
                review_citation_refs = []
            elif self.report_evidence_mode == "strict_subset":
                evidence = evidence[:-1]
                review_citation_refs = review_citation_refs[:-1]
            return self._json([{
                "report_class": "formal",
                "category": "character_drift",
                "metadata": {
                    "confirmed_candidate_id": self.candidate_id,
                    "final_outcome": "conflict",
                    "evidence_binding": "review_citations_v1",
                    "review_citation_refs": review_citation_refs,
                },
                "evidence": evidence,
            }])
        if method == "GET" and path.endswith("/review-clues"):
            return self._json({
                "items": [], "truncated": False,
                "unavailable_count": 0, "scan_limited": False,
            })
        if method == "GET" and path.endswith("/provisional-clues"):
            return self._json({"items": [], "truncated": False})
        raise AssertionError((method, path, request.url.query))

    def _candidate(self, baseline_public, setup):
        selector = setup.candidate_selector
        return {
            "id": self.candidate_id,
            "project_id": self.project_id,
            "source_run_id": self.baseline_run_id,
            "review_state": "pending",
            "reviewable": True,
            "source_verified": True,
            "support_bindings_status": "verified",
            "support_bindings_v1": {
                "schema_version": "character-support-bindings-v1",
                "index_version": "assertion-index-v1",
                "bindings": [{
                    "evidence_index": 0,
                    "support_id": "L1:A1",
                    "target": {"support_id": "L1:A1", "role": "target",
                               "start_offset": 0, "end_offset": 2},
                }],
            },
            "character_key": selector.character_key,
            "trait_type": selector.trait_type,
            "origin": "explicit_setting",
            "polarity": selector.polarity,
            "stability": selector.stability,
            "revision": 0,
            "comparison_key": "core_personality:restraint",
            "evidence": [{
                "input_id": _uuid(110),
                "document_id": self.baseline_runtime_id,
                "document_name": f"{baseline_public.document_id}.md",
                "document_version": 1,
                "content_sha256": baseline_public.sha256,
                "line_start": selector.source_line_start,
                "line_end": selector.source_line_end,
                "text": "林澈长期保持克制，不会公开发怒。",
            }],
        }


def test_mock_http_execution_produces_bound_conflict(tmp_path):
    bundle, _, _ = _write_bundle(tmp_path / "bundle")
    api = SuccessfulApi(bundle)
    with httpx.Client(
        base_url=_config().base_url,
        transport=httpx.MockTransport(api),
    ) as client:
        report = runner.execute_with_transport(
            bundle, _config(), client, sleeper=lambda _: None
        )
    assert report.phase == "completed"
    assert report.independent_project_count == 1
    prediction = report.prediction_artifact.prediction.predictions[0]
    assert (prediction.outcome, prediction.surface) == ("conflict", "formal_issue")
    assert [row.role for row in prediction.citations] == ["B", "C", "C"]


@pytest.mark.parametrize(
    "report_evidence_mode",
    ("empty", "strict_subset", "trace_extra_c"),
)
def test_report_evidence_and_trace_must_match_exactly(
    tmp_path, report_evidence_mode
):
    bundle, _, _ = _write_bundle(tmp_path / "bundle")
    api = SuccessfulApi(bundle, report_evidence_mode=report_evidence_mode)
    with httpx.Client(
        base_url=_config().base_url,
        transport=httpx.MockTransport(api),
    ) as client:
        report = runner.execute_with_transport(
            bundle, _config(), client, sleeper=lambda _: None
        )

    assert report.phase == "partial"
    prediction = report.prediction_artifact.prediction.predictions[0]
    assert prediction.outcome == "unassessed"
    assert prediction.citations == ()


def test_duplicate_exact_candidates_fail_closed_without_draft(tmp_path):
    bundle, _, _ = _write_bundle(tmp_path / "bundle")
    api = SuccessfulApi(bundle, duplicate=True)
    with httpx.Client(
        base_url=_config().base_url,
        transport=httpx.MockTransport(api),
    ) as client:
        report = runner.execute_with_transport(
            bundle, _config(), client, sleeper=lambda _: None
        )
    assert report.phase == "partial"
    prediction = report.prediction_artifact.prediction.predictions[0]
    assert prediction.outcome == "unassessed"
    assert prediction.citations == ()


def test_unique_selection_rejects_every_unselected_candidate(tmp_path):
    bundle, _, _ = _write_bundle(tmp_path / "bundle")
    api = SuccessfulApi(bundle, extra_unmatched=True)
    with httpx.Client(
        base_url=_config().base_url,
        transport=httpx.MockTransport(api),
    ) as client:
        report = runner.execute_with_transport(
            bundle, _config(), client, sleeper=lambda _: None
        )
    assert report.phase == "completed"
    assert api.decisions == [(_uuid(108), "reject"), (_uuid(105), "confirm")]


@pytest.mark.parametrize("mutation", ["baseline_as_current", "target_as_baseline", "broad"])
def test_setup_aware_evidence_roles_and_exact_anchor_boundaries(tmp_path, mutation):
    bundle, public_case, setup_case = _write_bundle(tmp_path / "bundle")
    baseline = setup_case.candidate_selector.source_document_id
    target = setup_case.target_document_id
    by_document = {row.document_id: row for row in public_case.documents}
    refs = [
        {"handle": "B01", "role": "B",
         "document_name": f"{baseline}.md", "line_start": 1, "line_end": 1},
        {"handle": "C01", "role": "C",
         "document_name": f"{target}.md", "line_start": 1, "line_end": 1},
        {"handle": "C02", "role": "C",
         "document_name": f"{target}.md", "line_start": 2, "line_end": 2},
    ]
    if mutation == "baseline_as_current":
        refs[1]["document_name"] = f"{baseline}.md"
    elif mutation == "target_as_baseline":
        refs[0]["document_name"] = f"{target}.md"
    else:
        refs[1]["line_end"] = 2
    trace = {
        "citation_refs_incomplete": False,
        "event_independence": "yes",
        "independent_event_citations": ["C01", "C02"],
        "citation_refs": refs,
    }
    assert runner._strict_runtime_refs(public_case, setup_case, trace) is None
    assert set(by_document) == {baseline, target}


def test_bundle_loader_reads_only_fixed_controls_and_manifest_documents(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(root)
    (root / "unrelated-private.json").write_text("do not open", encoding="utf-8")
    allowed = {
        "public-input.json", "author-setup.json", "freeze.json",
        *[f"{row.document_id}.md" for row in bundle.public.cases[0].documents],
    }
    original = Path.read_bytes

    def guarded(path):
        assert path.name in allowed
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    loaded = runner.load_public_bundle(root)
    assert loaded.public.dataset_id == bundle.public.dataset_id


def test_preflight_cli_has_no_remote_side_effect_and_refuses_overwrite(tmp_path, monkeypatch):
    bundle_root = tmp_path / "bundle"
    _write_bundle(bundle_root)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(_config().model_dump(mode="json")), encoding="utf-8"
    )
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        runner.httpx,
        "Client",
        lambda *_args, **_kwargs: pytest.fail("preflight must not open HTTP"),
    )
    argv = [
        "--bundle-root", str(bundle_root),
        "--run-config", str(config_path),
        "--output-json", str(output),
        "--preflight-only",
    ]
    assert runner.main(argv) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["phase"] == "preflight"
    assert runner.main(argv) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["phase"] == "preflight"


def test_cli_requires_explicit_provider_opt_in():
    with pytest.raises(SystemExit):
        runner.parse_args([
            "--bundle-root", "C:/bundle",
            "--run-config", "C:/config.json",
            "--output-json", "C:/report.json",
        ])


def test_setup_rejects_author_axis_on_api_unsupported_dimension(tmp_path):
    _, _, setup_case = _write_bundle(tmp_path / "bundle")
    payload = setup_case.model_dump(mode="json")
    payload["candidate_selector"]["trait_type"] = "preference"
    with pytest.raises(ValueError, match="author_axis_for_unsupported_dimension"):
        runner.PublicAuthorSetupCase.model_validate_json(json.dumps(payload))


def test_setup_requires_author_axis_for_supported_dimension(tmp_path):
    _, _, setup_case = _write_bundle(tmp_path / "bundle")
    payload = setup_case.model_dump(mode="json")
    payload["author_axis"] = None
    with pytest.raises(ValueError, match="author_axis_required_for_supported_dimension"):
        runner.PublicAuthorSetupCase.model_validate_json(json.dumps(payload))


def test_run_config_rejects_wrong_runner_source_hash(tmp_path):
    payload = _config().model_dump(mode="json")
    payload["expected_runner_source_sha256"] = "0" * 64
    path = tmp_path / "run-config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(runner.RunnerFailure, match="runner_source_hash_mismatch"):
        runner.load_run_config(path)


def test_runner_source_hash_is_canonical_explicit_code_manifest():
    manifest = runner.current_runner_code_manifest()

    assert tuple(row.path for row in manifest.files) == runner.RUNNER_CODE_PATHS
    assert runner.current_runner_source_sha256() == runner._canonical_model_sha(
        manifest
    )


@pytest.mark.parametrize("relative", runner.RUNNER_CODE_PATHS)
def test_any_runner_manifest_dependency_change_invalidates_config(
    tmp_path, monkeypatch, relative
):
    config = _config()
    path = tmp_path / "run-config.json"
    path.write_text(json.dumps(config.model_dump(mode="json")), encoding="utf-8")
    target = (runner.ROOT / Path(*relative.split("/"))).resolve()
    original = Path.read_bytes

    def changed_source(source_path):
        payload = original(source_path)
        if source_path.resolve() == target:
            return payload + b"\n# manifest mutation"
        return payload

    monkeypatch.setattr(Path, "read_bytes", changed_source)

    with pytest.raises(runner.RunnerFailure, match="runner_source_hash_mismatch"):
        runner.load_run_config(path)


def test_runtime_provenance_expectation_fails_before_project_create(tmp_path):
    bundle, _, _ = _write_bundle(tmp_path / "bundle")
    config = _config().model_copy(
        update={"expected_runtime_provenance_sha256": "0" * 64}
    )
    api = SuccessfulApi(bundle)
    with httpx.Client(
        base_url=config.base_url,
        transport=httpx.MockTransport(api),
    ) as client:
        with pytest.raises(
            runner.RunnerFailure, match="runtime_provenance_expectation_mismatch"
        ):
            runner.execute_with_transport(bundle, config, client)
    assert not hasattr(api, "created")


def _successful_report(tmp_path) -> runner.SealedHttpRunReport:
    bundle, _, _ = _write_bundle(tmp_path / "bundle")
    api = SuccessfulApi(bundle)
    with httpx.Client(
        base_url=_config().base_url,
        transport=httpx.MockTransport(api),
    ) as client:
        return runner.execute_with_transport(
            bundle, _config(), client, sleeper=lambda _: None
        )


@pytest.mark.parametrize(
    "tamper", ["config", "provenance", "runner", "execution", "receipt", "count"]
)
def test_report_tampering_fails_strict_binding(tmp_path, tamper):
    report = _successful_report(tmp_path)
    payload = report.model_dump(mode="json")
    if tamper == "config":
        payload["run_config"]["repeat_id"] = "copied-repeat"
    elif tamper == "provenance":
        payload["runtime_provenance_sha256"] = "f" * 64
    elif tamper == "runner":
        payload["runner_source_sha256"] = "d" * 64
    elif tamper == "execution":
        payload["execution_id"] = _uuid(999)
    elif tamper == "receipt":
        payload["project_receipts"][0]["project_id_sha256"] = "e" * 64
    else:
        payload["independent_project_count"] = 0
    with pytest.raises(ValueError):
        runner.SealedHttpRunReport.model_validate_json(json.dumps(payload))


def test_report_binds_execution_and_project_receipt(tmp_path):
    report = _successful_report(tmp_path)
    assert report.execution_id == _config().execution_id
    assert report.run_config_sha256 == runner._canonical_model_sha(_config())
    assert report.runner_source_sha256 == runner.current_runner_source_sha256()
    assert report.project_set_sha256 == runner._project_set_sha(
        report.project_receipts
    )
    assert report.independent_project_count == 1
