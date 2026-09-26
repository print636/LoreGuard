import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts import run_character_axis_direction_live as direction


_ISOLATION = {
    "schema_version": direction.ISOLATION_SCHEMA,
    "isolated_sqlite": True,
    "eval_db_id": "9aa9d84e-57da-4ae5-989c-cc23f9c9d8c6",
    "database_path_sha256": "c" * 64,
    "instance_id_sha256": "d" * 64,
}


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _args(**overrides):
    values = {
        "phase": "preflight", "preflight_only": False, "suite": "dev",
        "base_url": "http://127.0.0.1:8000", "run_timeout_seconds": 1.0,
        "allow_provider_call": False, "checkpoint_json": None,
        "author_decisions_json": None, "output_json": None,
        "expected_eval_db_id": None, "expected_eval_instance_id": None,
        "expected_eval_db_path": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _candidate(suite, key, *, polarity=None):
    target, selector = direction._target_selectors(suite)[key]
    name = target["source_document"]
    line_number = target["source_line"]
    line = suite.source.files[name].decode("utf-8").splitlines()[line_number - 1]
    raw_polarity = polarity or selector["polarity"]
    trait_key = f"actual_model_axis_{key}"
    return {
        "id": f"candidate-{key}", "project_id": "project-id",
        "source_run_id": "baseline-run", "source_verified": True,
        "review_state": "pending", "reviewable": True, "revision": 0,
        "character_key": selector["character_key"],
        "trait_type": selector["trait_type"], "trait_key": trait_key,
        "comparison_key": direction.stable_trait_identity(
            selector["trait_type"], trait_key, selector["key_object"] or ""
        ),
        "value": "sk-secret-raw-model-text-that-must-not-appear",
        "polarity": raw_polarity, "stability": selector["stability"],
        "origin": "explicit_setting",
        "evidence": [{
            "document_name": name, "line_start": line_number,
            "line_end": line_number, "text": line,
        }],
        "support_bindings_status": "verified",
        "support_bindings_v1": {
            "schema_version": "character-support-bindings-v1",
            "index_version": "assertion-index-v1",
            "bindings": [{
                "evidence_index": 0, "support_id": f"L{line_number}:A2",
                "target": {
                    "support_id": f"L{line_number}:A2",
                    "start_offset": 0, "end_offset": len(line), "role": "target",
                },
                "actor_anchor_id": None, "label_anchor_id": None,
                "scope_relation": "local", "context": [],
            }],
        },
    }


def _checkpoint(fixture, suite_name="dev", candidates=None):
    suite = fixture.suites[suite_name]
    candidates = candidates or {
        key: _candidate(suite, key) for key in direction._target_selectors(suite)
    }
    return {
        "schema_version": direction.CHECKPOINT_SCHEMA, "phase": "prepare",
        "suite": suite_name, "manifest_sha256": fixture.manifest_sha256,
        "source_manifest_sha256": fixture.source_manifest_sha256,
        "author_plan_sha256": suite.plan_sha256,
        "baseline_admitted": True,
        "code_state": {"git_head": "a" * 40, "worktree_clean": True},
        "runtime_provenance_sha256": "b" * 64,
        "evaluation_isolation": _ISOLATION,
        "project_id": "project-id", "baseline_run_id": "baseline-run",
        "candidate_inventory": {
            key: {
                "match_count": 1, "candidate_ids": [row["id"]],
                "candidate_revisions": {row["id"]: row["revision"]},
                "unique": True,
            }
            for key, row in candidates.items()
        },
    }


def _choices(suite, candidates, *, overrides=None):
    overrides = overrides or {}
    return {
        key: {
            "candidate_key": key, "candidate_id": row["id"],
            "decision": "confirm",
            "axis_alignment": "same" if direction._target_selectors(suite)[key][1][
                "approved_axis_key"
            ] else None,
            **overrides.get(key, {}),
        }
        for key, row in candidates.items()
    }


class _FakeAPI:
    def __init__(self, candidates):
        self.candidates = copy.deepcopy(candidates)
        self.axes = {}
        self.requests = []
        self.fail_on_post = None
        self.post_count = 0
        self.baseline_project_id = "project-id"

    def request(self, _client, method, path, route, **kwargs):
        self.requests.append((method, route, kwargs))
        if route == "baseline_status":
            return {
                "id": "baseline-run", "status": "completed",
                "project_id": self.baseline_project_id,
            }
        if route == "axis_list":
            return {
                "items": list(self.axes.values()), "total": len(self.axes),
                "limit": 100, "offset": 0,
            }
        if route == "axis_create":
            body = kwargs["json"]
            axis = {
                "id": f"axis-{len(self.axes) + 1}", "version": 1,
                "trait_type": "core_personality",
                "definition_sha256": direction._digest(direction._normalized(body["definition"])),
                "positive_proposition_sha256": _hash(body["positive_proposition"]),
            }
            self.axes[axis["id"]] = axis
            return copy.deepcopy(axis)
        if route in {"candidate_detail", "candidate_decision"}:
            key = next(
                key for key, row in self.candidates.items() if row["id"] in path
            )
            row = self.candidates[key]
            if route == "candidate_detail":
                return {**copy.deepcopy(row), "decisions": copy.deepcopy(row.get("decisions", []))}
            self.post_count += 1
            if self.fail_on_post == self.post_count:
                direction._fail("simulated_interruption", "candidate_review")
            body = kwargs["json"]
            idem = kwargs["headers"]["Idempotency-Key"]
            if row["review_state"] == "confirmed":
                if row["idempotency_key"] != idem or row["posted_body"] != body:
                    direction._fail("idempotency_key_conflict", "candidate_review")
                deduplicated = True
            else:
                assert row["revision"] == body["expected_revision"]
                row["review_state"] = "confirmed"
                row["reviewable"] = False
                row["revision"] += 1
                row["approved_axis_id"] = body.get("approved_axis_id")
                row["approved_axis_version"] = body.get("expected_axis_version")
                row["axis_alignment"] = body.get("axis_alignment")
                row["axis_polarity"] = (
                    direction._axis_polarity_from_choice(
                        row["polarity"], body.get("axis_alignment")
                    ) if body.get("approved_axis_id") else None
                )
                row["axis_positive_proposition_sha256"] = body.get(
                    "expected_axis_positive_proposition_sha256"
                )
                row["idempotency_key"] = idem
                row["posted_body"] = copy.deepcopy(body)
                row["decisions"] = [{
                    "id": f"review-{key}", "decision": "confirm",
                    "expected_revision": body["expected_revision"],
                    "axis_alignment": body.get("axis_alignment"),
                    "axis_polarity": row["axis_polarity"],
                    "axis_positive_proposition_sha256": row[
                        "axis_positive_proposition_sha256"
                    ],
                    "comment": body["comment"],
                }]
                deduplicated = False
            return {
                "candidate": copy.deepcopy(row),
                "decision_id": row["decisions"][0]["id"],
                "deduplicated": deduplicated,
            }
        if route == "health_postrun":
            return {"runtime_provenance": {}}
        raise AssertionError(route)


def _mock_confirm(monkeypatch, checkpoint, api):
    monkeypatch.setattr(
        direction, "_verify_live_isolation", lambda *_a, **_k: {
            **_ISOLATION, "fresh_for_prepare": False,
        },
    )
    monkeypatch.setattr(
        direction, "_service_preflight",
        lambda _client: (checkpoint["code_state"], "b" * 64),
    )
    monkeypatch.setattr(
        direction.legacy, "_runtime_provenance_digest", lambda _value: "b" * 64
    )
    monkeypatch.setattr(direction.legacy, "_run_summary", lambda *_a, **_k: {
        "runtime_provenance_sha256": "b" * 64,
    })
    monkeypatch.setattr(
        direction.legacy, "_baseline_admission", lambda _summary: {"admitted": True}
    )
    monkeypatch.setattr(direction.legacy, "_request", api.request)


def _confirm(fixture, checkpoint, choices):
    return direction.confirm(
        object(), fixture, checkpoint, choices, expected_isolation=_ISOLATION
    )


def test_preflight_verifies_both_source_suites_without_http_or_oracle_parse(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("network or oracle parse in preflight")

    monkeypatch.setattr(direction.httpx, "Client", forbidden)
    monkeypatch.setattr(direction.legacy, "_load_oracle", forbidden)
    report, exit_code = direction.run(_args(preflight_only=True))
    assert exit_code == 0
    assert report["preflight_verified"] is True
    assert report["blind_holdout"] is False
    assert report["quality_scored"] is False
    assert report["passed"] is False
    assert set(report["author_plan_sha256"]) == {"dev", "transfer"}


def test_plan_and_source_hashes_fail_closed_without_model_calls(tmp_path, monkeypatch):
    fixture_dir = tmp_path / "direction"
    shutil.copytree(direction.DATASET, fixture_dir)
    plan_path = fixture_dir / "dev" / "author-plan.json"
    plan_path.write_bytes(plan_path.read_bytes() + b" ")
    with pytest.raises(direction.legacy.SafeFailure, match="direction_plan_hash_mismatch"):
        direction.verify_fixture(fixture_dir)

    source_dir = tmp_path / "source"
    shutil.copytree(direction.SOURCE_DATASET, source_dir)
    story_path = source_dir / "dev" / "02-character-profiles.md"
    story_path.write_bytes(story_path.read_bytes() + b" ")
    with pytest.raises(direction.legacy.SafeFailure, match="fixture_file_hash_mismatch"):
        direction.verify_fixture(source_dataset=source_dir)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("network in unapproved prepare")

    monkeypatch.setattr(direction.httpx, "Client", forbidden)
    report, exit_code = direction.run(_args(
        phase="prepare", output_json="artifacts/direction-unapproved-test.json",
    ))
    assert exit_code == 1
    assert report["failure"]["code"] == "explicit_provider_call_and_output_required"


def test_default_or_nonlocal_base_url_fails_before_http_or_model(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("HTTP client created for unsafe URL")

    monkeypatch.setattr(direction.httpx, "Client", forbidden)
    for url in (
        "http://127.0.0.1:8000", "http://localhost:8000",
        "http://127.0.0.1", "http://example.com:8765",
        "http://127.0.0.1:8765/other", "http://127.0.0.1:8765/?token=secret",
    ):
        report, exit_code = direction.run(_args(
            phase="prepare", base_url=url, allow_provider_call=True,
            output_json="artifacts/unsafe-isolation-test.json",
        ))
        assert exit_code == 1
        assert report["failure"]["code"] == "isolation_base_url_invalid"
        assert "secret" not in json.dumps(report)
    assert direction._isolated_base_url("http://127.0.0.1:8765") == (
        "http://127.0.0.1:8765"
    )
    report, exit_code = direction.run(_args(
        phase="prepare", base_url="http://127.0.0.1:8765",
        allow_provider_call=True, output_json="artifacts/unsafe-isolation-test.json",
    ))
    assert exit_code == 1
    assert report["failure"]["code"] == "isolation_expectation_required"
    secret_path = str(Path.cwd() / "sk-secret-do-not-log.sqlite3")
    report, exit_code = direction.run(_args(
        phase="prepare", base_url="http://127.0.0.1:8765",
        allow_provider_call=True, output_json="artifacts/unsafe-isolation-test.json",
        expected_eval_db_id=_ISOLATION["eval_db_id"],
        expected_eval_instance_id="44737d93-c1e2-44a2-9295-90e54837f31b",
        expected_eval_db_path=secret_path,
    ))
    assert exit_code == 1
    assert report["failure"]["code"] == "isolation_expectation_invalid"
    assert "sk-secret" not in json.dumps(report)
    assert "DATABASE_URL" not in json.dumps(report)


def test_live_isolation_rejects_wrong_ids_and_nonfresh_prepare(monkeypatch):
    observed = {**_ISOLATION, "fresh_for_prepare": False}
    monkeypatch.setattr(
        direction.legacy, "_request", lambda *_a, **_k: observed,
    )
    with pytest.raises(direction.legacy.SafeFailure, match="evaluation_isolation_mismatch"):
        direction._verify_live_isolation(
            object(), _ISOLATION, require_fresh=True, stage="isolation_preflight"
        )
    assert direction._verify_live_isolation(
        object(), _ISOLATION, require_fresh=False, stage="isolation_preflight"
    ) == observed
    for key in ("eval_db_id", "database_path_sha256", "instance_id_sha256"):
        wrong = {**_ISOLATION, key: "wrong"}
        with pytest.raises(direction.legacy.SafeFailure, match="evaluation_isolation_mismatch"):
            direction._verify_live_isolation(
                object(), wrong, require_fresh=False, stage="isolation_preflight"
            )


def test_core_selector_uses_verified_target_but_not_old_raw_polarity():
    suite = direction.verify_fixture().suites["dev"]
    target, selector = direction._target_selectors(suite)["dev_partner_signature"]
    assert selector["polarity"] == "negative"
    candidate = _candidate(suite, "dev_partner_signature", polarity="positive")
    assert direction._matches_target(
        candidate, target, selector, suite,
        project_id="project-id", baseline_run_id="baseline-run",
    )
    wrong_target = copy.deepcopy(candidate)
    wrong_target["support_bindings_v1"]["bindings"][0]["target"]["end_offset"] = 4
    assert not direction._matches_target(
        wrong_target, target, selector, suite,
        project_id="project-id", baseline_run_id="baseline-run",
    )
    wrong_actor = {**candidate, "character_key": "another actor"}
    assert not direction._matches_target(
        wrong_actor, target, selector, suite,
        project_id="project-id", baseline_run_id="baseline-run",
    )


def test_prepare_requires_precise_scope_review_before_project_or_model_call(monkeypatch):
    monkeypatch.setattr(direction.legacy, "_code_state", lambda: {
        "git_head": "a" * 40, "worktree_clean": True,
    })
    monkeypatch.setattr(
        direction.legacy, "_local_service_artifact_sha256", lambda _root: "c" * 64
    )
    monkeypatch.setattr(direction.legacy, "_service_preflight_gate", lambda *_a: "b" * 64)
    monkeypatch.setattr(
        direction.legacy, "_safe_character_runtime_provenance",
        lambda _value: {"character_consistency_limits": {}},
    )
    routes = []

    def fake_request(_client, _method, _path, route, **_kwargs):
        routes.append(route)
        if route == "health":
            return {"runtime_provenance": {}}
        pytest.fail(f"unexpected HTTP route: {route}")

    monkeypatch.setattr(direction.legacy, "_request", fake_request)
    with pytest.raises(direction.legacy.SafeFailure, match="precise_scope_review_not_enabled"):
        direction._service_preflight(object())
    assert routes == ["health"]


def test_author_decision_file_requires_explicit_choice_and_declaration(tmp_path):
    fixture = direction.verify_fixture()
    suite = fixture.suites["dev"]
    checkpoint = _checkpoint(fixture)
    choices = []
    for key, (_, selector) in direction._target_selectors(suite).items():
        choices.append({
            "candidate_key": key, "candidate_id": f"candidate-{key}",
            "decision": "confirm",
            "axis_alignment": "same" if selector["approved_axis_key"] else None,
        })
    document = {
        "schema_version": direction.DECISIONS_SCHEMA,
        "project_id": checkpoint["project_id"],
        "baseline_run_id": checkpoint["baseline_run_id"],
        "reviewer_declaration": direction.HUMAN_DECLARATION,
        "choices": choices,
    }
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert len(direction._load_decisions(path, checkpoint, suite)) == 5

    document["choices"][3]["axis_alignment"] = None
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(direction.legacy.SafeFailure, match="author_alignment_contract"):
        direction._load_decisions(path, checkpoint, suite)

    document["choices"][3]["axis_alignment"] = "opposite"
    document["reviewer_declaration"] = "generated from oracle"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(direction.legacy.SafeFailure, match="author_decisions_contract"):
        direction._load_decisions(path, checkpoint, suite)


def test_prepare_uploads_background_only_and_freezes_safe_provenance(monkeypatch):
    fixture = direction.verify_fixture()
    suite = fixture.suites["dev"]
    uploads = []
    monkeypatch.setattr(
        direction, "_service_preflight",
        lambda _client: ({"git_head": "a" * 40, "worktree_clean": True}, "b" * 64),
    )
    monkeypatch.setattr(direction.legacy, "_runtime_provenance_digest", lambda _value: "b" * 64)
    monkeypatch.setattr(direction.legacy, "_request", lambda _c, _m, _p, route, **_k: (
        {"id": "project-id"} if route == "project_create"
        else {"runtime_provenance": {}} if route == "health_postrun"
        else pytest.fail(f"unexpected route {route}")
    ))
    monkeypatch.setattr(direction.legacy, "_upload", lambda _c, _p, _s, name, _role, **_k: (
        uploads.append(name) or f"document-{name}"
    ))
    monkeypatch.setattr(direction.legacy, "_start_run", lambda *_a, **_k: "baseline-run")
    monkeypatch.setattr(direction.legacy, "_wait_run", lambda *_a, **_k: {"id": "baseline-run"})
    monkeypatch.setattr(direction.legacy, "_run_summary", lambda *_a, **_k: {
        "run_id": "baseline-run", "runtime_provenance_sha256": "b" * 64,
        "status": "completed", "stage_outcome": "completed",
        "material_coverage": "complete", "planned_chunks": 1,
        "processed_chunks": 1, "stage_usage": {"attempted_calls": 1},
        "case_trace": [], "visible_issue_cases": [],
    })
    monkeypatch.setattr(direction.legacy, "_baseline_admission", lambda _summary: {
        "admitted": True,
    })
    monkeypatch.setattr(direction.legacy, "_list_pending", lambda _c, _p: [
        _candidate(suite, key) for key in direction._target_selectors(suite)
    ])
    isolation_calls = []
    monkeypatch.setattr(
        direction, "_verify_live_isolation",
        lambda *_a, **kwargs: isolation_calls.append(kwargs["require_fresh"]) or {
            **_ISOLATION, "fresh_for_prepare": kwargs["require_fresh"],
        },
    )
    report = direction.prepare(
        object(), fixture, "dev", timeout_seconds=1,
        expected_isolation=_ISOLATION,
    )
    assert uploads == [name for name, _ in direction.legacy.BASELINE_FILES]
    assert direction.legacy.DRAFT_FILE not in uploads
    assert report["baseline_admitted"] is True
    assert report["runtime_provenance_sha256"] == "b" * 64
    assert report["author_review_required"] is True
    assert report["quality_scored"] is False
    assert report["evaluation_isolation"] == _ISOLATION
    assert isolation_calls == [True, False]
    assert all(row["unique"] for row in report["candidate_inventory"].values())
    rendered = json.dumps(report, ensure_ascii=False)
    assert "sk-secret" not in rendered
    assert "桑衍" not in rendered


@pytest.mark.parametrize(
    ("signature_alignment", "expected_agreement"),
    [("opposite", True), ("same", False)],
)
def test_explicit_choice_is_sent_and_scored_without_gold_signing(
    monkeypatch, signature_alignment, expected_agreement,
):
    fixture = direction.verify_fixture()
    suite = fixture.suites["dev"]
    candidates = {
        key: _candidate(
            suite, key,
            polarity="positive" if key in {"dev_partner_signature", "dev_shift_error"} else None,
        )
        for key in direction._target_selectors(suite)
    }
    choices = _choices(suite, candidates, overrides={
        "dev_shift_error": {"axis_alignment": "opposite"},
        "dev_partner_signature": {"axis_alignment": signature_alignment},
    })
    checkpoint = _checkpoint(fixture, candidates=candidates)
    api = _FakeAPI(candidates)
    _mock_confirm(monkeypatch, checkpoint, api)
    report = _confirm(fixture, checkpoint, choices)
    assert report["api_decision_integrity"] is True
    assert report["plan_agreement"]["all_compared_match"] is expected_agreement
    assert report["coverage"] == {"targets_total": 5, "confirmed": 5, "deferred": 0}
    assert report["quality_scored"] is False
    assert report["passed"] is False
    assert report["author_review_provenance"] == (
        "decision_file_self_attested_not_independently_verified"
    )
    decision_bodies = [
        kwargs["json"] for _, route, kwargs in api.requests if route == "candidate_decision"
    ]
    assert len(decision_bodies) == 5
    assert sum(body.get("axis_alignment") == "opposite" for body in decision_bodies) == (
        2 if signature_alignment == "opposite" else 1
    )
    assert all(
        body.get("expected_axis_positive_proposition_sha256")
        for body in decision_bodies if body.get("approved_axis_id")
    )
    rendered = json.dumps(report, ensure_ascii=False)
    assert "sk-secret" not in rendered
    assert "不会冒用搭档签名" not in rendered
    assert "桑衍" not in rendered


def test_retry_after_partial_confirmation_replays_exact_stable_decision(monkeypatch):
    fixture = direction.verify_fixture()
    suite = fixture.suites["dev"]
    candidates = {key: _candidate(suite, key) for key in direction._target_selectors(suite)}
    checkpoint = _checkpoint(fixture, candidates=candidates)
    choices = _choices(suite, candidates)
    api = _FakeAPI(candidates)
    api.fail_on_post = 2
    _mock_confirm(monkeypatch, checkpoint, api)
    with pytest.raises(direction.legacy.SafeFailure, match="simulated_interruption"):
        _confirm(fixture, checkpoint, choices)
    first_post = next(
        kwargs for _, route, kwargs in api.requests if route == "candidate_decision"
    )
    api.fail_on_post = None
    report = _confirm(fixture, checkpoint, choices)
    assert report["api_decision_integrity"] is True
    assert report["coverage"]["confirmed"] == 5
    retry_posts = [
        kwargs for _, route, kwargs in api.requests if route == "candidate_decision"
    ][2:]
    assert retry_posts[0]["headers"]["Idempotency-Key"] == first_post[
        "headers"
    ]["Idempotency-Key"]
    assert retry_posts[0]["json"] == first_post["json"]
    assert len(api.axes) == 4


def test_replay_fails_closed_if_review_chain_changed(monkeypatch):
    fixture = direction.verify_fixture()
    suite = fixture.suites["dev"]
    candidates = {key: _candidate(suite, key) for key in direction._target_selectors(suite)}
    checkpoint = _checkpoint(fixture, candidates=candidates)
    choices = _choices(suite, candidates)
    api = _FakeAPI(candidates)
    _mock_confirm(monkeypatch, checkpoint, api)
    _confirm(fixture, checkpoint, choices)
    api.candidates["dev_partner_signature"]["decisions"][0]["axis_alignment"] = "opposite"
    api.requests.clear()
    with pytest.raises(
        direction.legacy.SafeFailure, match="confirmed_candidate_review_chain_mismatch"
    ):
        _confirm(fixture, checkpoint, choices)
    assert not any(route in {"axis_create", "candidate_decision"} for _, route, _ in api.requests)


def test_replay_fails_closed_if_axis_mapping_changed(monkeypatch):
    fixture = direction.verify_fixture()
    suite = fixture.suites["dev"]
    candidates = {key: _candidate(suite, key) for key in direction._target_selectors(suite)}
    checkpoint = _checkpoint(fixture, candidates=candidates)
    choices = _choices(suite, candidates)
    api = _FakeAPI(candidates)
    _mock_confirm(monkeypatch, checkpoint, api)
    _confirm(fixture, checkpoint, choices)
    api.candidates["dev_partner_signature"]["approved_axis_id"] = "unrelated-axis"
    api.requests.clear()
    with pytest.raises(direction.legacy.SafeFailure, match="axis_create_contract"):
        _confirm(fixture, checkpoint, choices)
    assert not any(route in {"axis_create", "candidate_decision"} for _, route, _ in api.requests)


@pytest.mark.parametrize("defer_all", [False, True])
def test_defer_is_integrity_success_and_creates_no_unauthorized_axis(
    monkeypatch, defer_all,
):
    fixture = direction.verify_fixture()
    suite = fixture.suites["dev"]
    candidates = {key: _candidate(suite, key) for key in direction._target_selectors(suite)}
    checkpoint = _checkpoint(fixture, candidates=candidates)
    deferred = list(candidates) if defer_all else ["dev_partner_signature"]
    choices = _choices(suite, candidates, overrides={
        key: {"decision": "defer", "axis_alignment": "uncertain"} for key in deferred
    })
    api = _FakeAPI(candidates)
    _mock_confirm(monkeypatch, checkpoint, api)
    report = _confirm(fixture, checkpoint, choices)
    assert report["api_decision_integrity"] is True
    assert report["coverage"]["deferred"] == len(deferred)
    assert report["coverage"]["confirmed"] == 5 - len(deferred)
    assert len(api.axes) == (0 if defer_all else 3)
    assert sum(route == "candidate_decision" for _, route, _ in api.requests) == 5 - len(deferred)
    assert all(api.candidates[key]["review_state"] == "pending" for key in deferred)
    if defer_all:
        assert not any(route == "axis_list" for _, route, _ in api.requests)


def test_baseline_project_and_checkpoint_candidate_binding_fail_before_write(monkeypatch):
    fixture = direction.verify_fixture()
    suite = fixture.suites["dev"]
    candidates = {key: _candidate(suite, key) for key in direction._target_selectors(suite)}
    checkpoint = _checkpoint(fixture, candidates=candidates)
    choices = _choices(suite, candidates)
    api = _FakeAPI(candidates)
    _mock_confirm(monkeypatch, checkpoint, api)
    api.baseline_project_id = "other-project"
    with pytest.raises(direction.legacy.SafeFailure, match="baseline_status_changed"):
        _confirm(fixture, checkpoint, choices)
    assert not any(route in {"axis_create", "candidate_decision"} for _, route, _ in api.requests)
    api.baseline_project_id = "project-id"
    api.requests.clear()
    checkpoint["candidate_inventory"]["dev_partner_signature"]["candidate_ids"] = []
    with pytest.raises(direction.legacy.SafeFailure, match="checkpoint_candidate_mismatch"):
        _confirm(fixture, checkpoint, choices)
    assert not any(route in {"axis_create", "candidate_decision"} for _, route, _ in api.requests)


def test_confirm_rejects_changed_isolation_checkpoint_before_http(monkeypatch):
    fixture = direction.verify_fixture()
    suite = fixture.suites["dev"]
    candidates = {key: _candidate(suite, key) for key in direction._target_selectors(suite)}
    checkpoint = _checkpoint(fixture, candidates=candidates)
    checkpoint["evaluation_isolation"] = {
        **_ISOLATION, "eval_db_id": "2d1dbff1-d885-4f9e-9767-447d94668226",
    }
    choices = _choices(suite, candidates)
    monkeypatch.setattr(
        direction.legacy, "_request", lambda *_a, **_k: pytest.fail("unexpected HTTP")
    )
    with pytest.raises(direction.legacy.SafeFailure, match="checkpoint_isolation_mismatch"):
        _confirm(fixture, checkpoint, choices)


def test_cli_confirm_exits_zero_for_valid_defer_and_plan_disagreement(
    tmp_path, monkeypatch,
):
    fixture = direction.verify_fixture()
    suite = fixture.suites["dev"]
    candidates = {
        key: _candidate(suite, key, polarity="positive")
        for key in direction._target_selectors(suite)
    }
    checkpoint = _checkpoint(fixture, candidates=candidates)
    choices = _choices(suite, candidates, overrides={
        "dev_partner_signature": {"axis_alignment": "same"},
        "dev_shift_error": {"decision": "defer", "axis_alignment": "uncertain"},
    })
    api = _FakeAPI(candidates)
    _mock_confirm(monkeypatch, checkpoint, api)
    monkeypatch.setattr(direction, "verify_fixture", lambda: fixture)
    monkeypatch.setattr(direction, "_json_file", lambda *_a, **_k: checkpoint)
    monkeypatch.setattr(direction, "_load_decisions", lambda *_a, **_k: choices)
    monkeypatch.setattr(
        direction.legacy, "_resolve_output_json", lambda name: tmp_path / Path(name).name
    )

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(direction.httpx, "Client", lambda **_kwargs: _Client())
    monkeypatch.setattr(direction, "_expected_isolation", lambda *_a: _ISOLATION)
    report, exit_code = direction.run(_args(
        phase="confirm", checkpoint_json="checkpoint.json",
        author_decisions_json="decisions.json", output_json="result.json",
        base_url="http://127.0.0.1:8765",
    ))
    assert exit_code == 0
    assert report["api_decision_integrity"] is True
    assert report["coverage"]["deferred"] == 1
    assert report["plan_agreement"]["all_compared_match"] is False
