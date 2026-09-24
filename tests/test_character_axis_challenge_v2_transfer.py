"""Check v2 transfer labels against their independent source world.

These checks validate authored anchors; they do not measure model quality.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "character-axis-challenge-v2"
    / "transfer"
)
PROFILE = "02-character-profiles.md"
HISTORY = "03-published-history-v1.0.md"
DRAFT = "04-draft-event-v1.1.md"
DOCUMENTS = {
    "01-world-setting.md",
    PROFILE,
    HISTORY,
    DRAFT,
}


def _json(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _line(name: str, number: int) -> str:
    assert name in DOCUMENTS
    assert type(number) is int and number > 0
    lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
    assert number <= len(lines)
    value = lines[number - 1]
    assert value.strip()
    return value


def test_transfer_world_has_bounded_original_content_and_schema() -> None:
    assert {path.name for path in ROOT.iterdir() if path.is_file()} == (
        DOCUMENTS | {"review-plan.json", "oracle.json"}
    )
    hanzi = sum(
        len(re.findall(r"[\u3400-\u9fff]", (ROOT / name).read_text(encoding="utf-8")))
        for name in DOCUMENTS
    )
    assert 1500 <= hanzi <= 2500

    plan = _json("review-plan.json")
    oracle = _json("oracle.json")
    assert set(plan) == {
        "schema_version", "suite", "world_id", "approved_axes", "candidate_decisions"
    }
    assert plan["schema_version"] == "character-axis-review-plan-v1"
    assert plan["suite"] == "transfer"
    assert set(oracle) == {
        "schema_version", "suite", "world_id", "developer_visible", "expected_cases"
    }
    assert oracle["schema_version"] == "character-axis-oracle-v1"
    assert oracle["suite"] == "transfer"
    assert oracle["world_id"] == plan["world_id"]
    assert oracle["developer_visible"] is True
    assert len(plan["candidate_decisions"]) == len(oracle["expected_cases"]) == 6
    assert Counter(case["gold_class"] for case in oracle["expected_cases"]) == {
        "conflict": 2, "explained": 2, "hard_negative": 2,
    }


def test_transfer_selectors_and_case_evidence_are_line_grounded() -> None:
    plan = _json("review-plan.json")
    oracle = _json("oracle.json")
    axes = {axis["axis_key"]: axis for axis in plan["approved_axes"]}
    assert len(axes) == len(plan["approved_axes"]) == 5
    for axis in axes.values():
        assert set(axis) == {"axis_key", "display_name", "definition", "trait_type"}
        assert axis["trait_type"] == "core_personality"
        assert axis["display_name"] and axis["definition"]

    candidates = {row["candidate_key"]: row for row in plan["candidate_decisions"]}
    assert len(candidates) == 6
    assert "04-draft-event-v1.1.md" not in json.dumps(plan, ensure_ascii=False)
    assert "gold_class" not in json.dumps(plan, ensure_ascii=False)
    for candidate in candidates.values():
        assert set(candidate) == {
            "candidate_key", "character_key", "trait_type", "source_document",
            "source_line", "source_quote", "polarity", "stability",
            "key_object", "approved_axis_key",
        }
        assert candidate["source_document"] == PROFILE
        source = _line(PROFILE, candidate["source_line"])
        assert candidate["character_key"] in source
        assert candidate["source_quote"] in source
        assert candidate["polarity"] in {"positive", "negative"}
        if candidate["trait_type"] == "core_personality":
            assert "核心人格" in source
            assert candidate["stability"] == "core"
            assert candidate["approved_axis_key"] in axes
            assert candidate["key_object"] is None
        else:
            assert candidate["trait_type"] == "preference"
            assert "长期稳定偏好" in source
            assert candidate["stability"] == "stable"
            assert candidate["approved_axis_key"] is None
            assert candidate["key_object"] in source

    ids = [case["case_id"] for case in oracle["expected_cases"]]
    assert len(set(ids)) == len(ids)
    assert {case["candidate_key"] for case in oracle["expected_cases"]} == set(candidates)
    for case in oracle["expected_cases"]:
        candidate = candidates[case["candidate_key"]]
        assert set(case) == {
            "case_id", "candidate_key", "gold_class", "allowed_final_outcomes",
            "required_citation_roles", "min_independent_observations", "evidence", "notes",
        }
        evidence = case["evidence"]
        assert set(evidence) == {"B", "C", "G", "X", "forbidden"}
        assert evidence["B"] == [{
            "document_name": PROFILE,
            "line": candidate["source_line"],
        }]
        for role in ("B", "C", "G", "X", "forbidden"):
            for ref in evidence[role]:
                assert set(ref) == (
                    {"document_name", "line", "character_key", "role", "polarity"}
                    if role == "forbidden" else {"document_name", "line"}
                )
                assert ref["document_name"] == (
                    PROFILE if role == "B" else DRAFT if role in {"C", "forbidden"}
                    else HISTORY
                )
                source = _line(ref["document_name"], ref["line"])
                if role == "C":
                    assert candidate["character_key"] in source
                if role == "forbidden":
                    assert ref["character_key"] == candidate["character_key"]
                    assert ref["role"] == "C"
                    assert ref["polarity"] in {"positive", "negative", None}
        assert len({ref["line"] for ref in evidence["C"]}) == len(evidence["C"])
        assert case["min_independent_observations"] <= len(evidence["C"])
        if case["gold_class"] == "conflict":
            assert case["allowed_final_outcomes"] == ["conflict"]
            assert case["min_independent_observations"] == 2
        elif case["gold_class"] == "explained":
            assert case["allowed_final_outcomes"] == ["no_issue"]
            assert evidence["G"] or evidence["X"]
        else:
            assert "conflict" not in case["allowed_final_outcomes"]
            assert evidence["forbidden"]


def test_transfer_medical_bridge_and_forbidden_lines_are_unambiguous() -> None:
    cases = {case["case_id"]: case for case in _json("oracle.json")["expected_cases"]}
    medical = cases["v2t_medicinal_preference_exception"]
    bridge_ref = medical["evidence"]["X"][0]
    bridge = _line(bridge_ref["document_name"], bridge_ref["line"])
    assert bridge.index("医师在治疗记录") < bridge.index("童画当天就依医嘱")
    assert bridge.index("童画当天就依医嘱") < bridge.index("记录没有说她不再喜欢")
    assert "热椒饼" in bridge and "三天内" in bridge

    wrong_actor = cases["v2t_other_actor_copies_interview"]
    forbidden = wrong_actor["evidence"]["forbidden"]
    assert len(forbidden) == 1
    offending = _line(forbidden[0]["document_name"], forbidden[0]["line"])
    assert "苏栈" in offending and "罗月" in offending and "复制" in offending
    assert "阻止" not in offending and forbidden[0]["polarity"] == "positive"
    lawful = _line(DRAFT, forbidden[0]["line"] + 2)
    assert "罗月" in lawful and "阻止" in lawful and "没有参与复制" in lawful

    quote_case = cases["v2t_unplayed_branch_quote"]
    quote_ref = quote_case["evidence"]["forbidden"][0]
    quote = _line(quote_ref["document_name"], quote_ref["line"])
    assert quote_ref["polarity"] is None
    assert "废弃分支设想" in quote and "如果杭泊" in quote
    assert "没有角色实际选择该分支" in quote
