"""Validate the preregistered, developer-visible character-axis fixture only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data" / "character-axis-challenge-v1"
SUITES = ("dev", "transfer")
FILES = frozenset(
    {
        "01-world-setting.md",
        "02-character-profiles.md",
        "03-published-history-v1.0.md",
        "04-draft-event-v1.1.md",
        "review-plan.json",
        "oracle.json",
    }
)
PLAN_KEYS = frozenset(
    {"schema_version", "suite", "world_id", "approved_axes", "candidate_decisions"}
)
CANDIDATE_KEYS = frozenset(
    {
        "candidate_key", "character_key", "trait_type", "source_document",
        "source_line", "source_quote", "polarity", "stability", "key_object",
        "approved_axis_key",
    }
)
CASE_KEYS = frozenset(
    {
        "case_id", "candidate_key", "gold_class", "allowed_final_outcomes",
        "required_citation_roles", "min_independent_observations", "evidence", "notes",
    }
)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _line(root: Path, reference: dict) -> str:
    assert set(reference) >= {"document_name", "line"}
    name = reference["document_name"]
    assert name in FILES and name.endswith(".md")
    line_number = reference["line"]
    assert type(line_number) is int and line_number > 0
    lines = (root / name).read_text(encoding="utf-8").splitlines()
    assert line_number <= len(lines)
    assert lines[line_number - 1].strip()
    return lines[line_number - 1]


def test_manifest_freezes_each_input_and_label_file() -> None:
    manifest = _json(ROOT / "manifest.json")
    assert set(manifest) == {"schema_version", "dataset_boundary", "suites"}
    assert manifest["schema_version"] == "character-axis-fixture-manifest-v1"
    assert manifest["dataset_boundary"] == {
        "developer_visible": True,
        "blind_holdout": False,
        "production_quality": False,
    }
    assert set(manifest["suites"]) == set(SUITES)
    for suite, entry in manifest["suites"].items():
        root = ROOT / suite
        assert set(entry) == {"world_id", "case_count", "files"}
        assert set(entry["files"]) == FILES
        assert {path.name for path in root.iterdir() if path.is_file()} == FILES
        assert entry["case_count"] >= 4
        for name, expected_hash in entry["files"].items():
            assert len(expected_hash) == 64
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected_hash
        assert len(_json(root / "oracle.json")["expected_cases"]) == entry["case_count"]


def test_review_plans_are_source_grounded_and_contain_no_draft_answers() -> None:
    worlds: set[str] = set()
    people: set[str] = set()
    axis_keys: set[str] = set()
    candidate_keys: set[str] = set()
    for suite in SUITES:
        root = ROOT / suite
        plan = _json(root / "review-plan.json")
        assert set(plan) == PLAN_KEYS
        assert plan["schema_version"] == "character-axis-review-plan-v1"
        assert plan["suite"] == suite
        assert plan["world_id"] not in worlds
        worlds.add(plan["world_id"])
        serialized = json.dumps(plan, ensure_ascii=False)
        assert "trait_key" not in serialized
        assert "gold_class" not in serialized
        assert "04-draft-event-v1.1.md" not in serialized
        axes = {axis["axis_key"]: axis for axis in plan["approved_axes"]}
        assert len(axes) == len(plan["approved_axes"])
        assert not axis_keys.intersection(axes)
        axis_keys.update(axes)
        for axis in axes.values():
            assert set(axis) == {"axis_key", "display_name", "definition", "trait_type"}
            assert axis["trait_type"] == "core_personality"
            assert axis["display_name"] and axis["definition"]
        keys_here: set[str] = set()
        for candidate in plan["candidate_decisions"]:
            assert set(candidate) == CANDIDATE_KEYS
            assert candidate["source_document"] == "02-character-profiles.md"
            assert candidate["source_quote"] in _line(
                root,
                {
                    "document_name": candidate["source_document"],
                    "line": candidate["source_line"],
                },
            )
            assert candidate["character_key"] in _line(
                root,
                {
                    "document_name": candidate["source_document"],
                    "line": candidate["source_line"],
                },
            )
            assert candidate["polarity"] in {"positive", "negative"}
            assert candidate["stability"] in {"core", "stable"}
            if candidate["trait_type"] == "core_personality":
                assert candidate["approved_axis_key"] in axes
                assert candidate["stability"] == "core"
            else:
                assert candidate["approved_axis_key"] is None
                assert candidate["key_object"]
            assert candidate["candidate_key"] not in keys_here
            keys_here.add(candidate["candidate_key"])
            people.add(candidate["character_key"])
        assert not candidate_keys.intersection(keys_here)
        candidate_keys.update(keys_here)
    assert len(worlds) == 2
    assert len(people) >= 7


def test_oracles_bind_cases_to_exact_source_lines_and_safe_outcomes() -> None:
    case_ids: set[str] = set()
    classes: set[str] = set()
    for suite in SUITES:
        root = ROOT / suite
        plan = _json(root / "review-plan.json")
        oracle = _json(root / "oracle.json")
        assert set(oracle) == {
            "schema_version", "suite", "world_id", "developer_visible", "expected_cases"
        }
        assert oracle["schema_version"] == "character-axis-oracle-v1"
        assert oracle["suite"] == suite
        assert oracle["world_id"] == plan["world_id"]
        assert oracle["developer_visible"] is True
        candidates = {row["candidate_key"]: row for row in plan["candidate_decisions"]}
        assert len(candidates) == len(oracle["expected_cases"])
        used_candidates: set[str] = set()
        for case in oracle["expected_cases"]:
            assert set(case) == CASE_KEYS
            assert case["case_id"] not in case_ids
            case_ids.add(case["case_id"])
            key = case["candidate_key"]
            assert key in candidates and key not in used_candidates
            used_candidates.add(key)
            assert case["notes"]
            gold = case["gold_class"]
            assert gold in {"conflict", "explained", "hard_negative", "abstain"}
            classes.add(gold)
            outcomes = case["allowed_final_outcomes"]
            assert outcomes and len(outcomes) == len(set(outcomes))
            assert set(outcomes) <= {
                "conflict", "no_issue", "needs_confirmation", "unverifiable"
            }
            evidence = case["evidence"]
            assert set(evidence) == {"B", "C", "G", "X", "forbidden"}
            assert evidence["B"] == [{
                "document_name": candidates[key]["source_document"],
                "line": candidates[key]["source_line"],
            }]
            for role in ("B", "C", "G", "X"):
                for ref in evidence[role]:
                    source_line = _line(root, ref)
                    expected_document = (
                        "02-character-profiles.md" if role == "B" else
                        "04-draft-event-v1.1.md" if role == "C" else
                        "03-published-history-v1.0.md"
                    )
                    assert ref["document_name"] == expected_document
                    if role == "C":
                        assert candidates[key]["character_key"] in source_line
            for ref in evidence["forbidden"]:
                assert set(ref) == {
                    "document_name", "line", "character_key", "role", "polarity"
                }
                _line(root, ref)
                assert ref["document_name"] == "04-draft-event-v1.1.md"
                assert ref["character_key"] == candidates[key]["character_key"]
                assert ref["role"] == "C"
                assert ref["polarity"] in {"positive", "negative", None}
            minimum = case["min_independent_observations"]
            assert type(minimum) is int and 0 <= minimum <= len(evidence["C"])
            assert len({ref["line"] for ref in evidence["C"]}) == len(evidence["C"])
            roles = case["required_citation_roles"]
            assert len(roles) == len(set(roles)) and set(roles) <= {"B", "C", "G", "X"}
            if gold == "conflict":
                assert outcomes == ["conflict"] and minimum >= 2
                assert set(roles) == {"B", "C"}
            elif gold == "explained":
                assert outcomes == ["no_issue"] and minimum >= 1
                assert evidence["G"] or evidence["X"]
                assert {"B", "C"} < set(roles)
            elif gold == "abstain":
                assert "conflict" not in outcomes and "no_issue" not in outcomes
                assert evidence["forbidden"]
            else:
                assert "conflict" not in outcomes
    assert classes == {"conflict", "explained", "hard_negative", "abstain"}
    assert len(case_ids) == 10


def test_forbidden_actor_attribution_catches_any_direction_where_required() -> None:
    expected_null = {
        "dev": set(),
        "transfer": {
            ("transfer_ambiguous_seal_actor", 10),
            ("transfer_unplayed_branch_quote", 13),
        },
    }
    for suite, cases in expected_null.items():
        oracle = _json(ROOT / suite / "oracle.json")
        actual = {
            (case["case_id"], ref["line"])
            for case in oracle["expected_cases"]
            for ref in case["evidence"]["forbidden"]
            if ref["polarity"] is None
        }
        assert actual == cases
    dev = _json(ROOT / "dev" / "oracle.json")
    transfer = _json(ROOT / "transfer" / "oracle.json")
    assert next(
        ref["polarity"]
        for case in dev["expected_cases"]
        if case["case_id"] == "dev_other_actor_opens_seal"
        for ref in case["evidence"]["forbidden"]
        if ref["line"] == 5
    ) == "positive"
    assert next(
        ref["polarity"]
        for case in dev["expected_cases"]
        if case["case_id"] == "dev_other_actor_opens_seal"
        for ref in case["evidence"]["forbidden"]
        if ref["line"] == 6
    ) == "positive"
    assert next(
        ref["polarity"]
        for case in transfer["expected_cases"]
        if case["case_id"] == "transfer_ambiguous_seal_actor"
        for ref in case["evidence"]["forbidden"]
        if ref["line"] == 9
    ) == "positive"
