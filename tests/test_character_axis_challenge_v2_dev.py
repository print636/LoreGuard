"""Keep the unfrozen v2 DEV story and its preregistered anchors in sync."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data" / "character-axis-challenge-v2" / "dev"


def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _line(name: str, number: int) -> str:
    lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
    assert type(number) is int and 0 < number <= len(lines)
    assert lines[number - 1].strip()
    return lines[number - 1]


def _cases() -> dict[str, dict]:
    return {case["case_id"]: case for case in _load("oracle.json")["expected_cases"]}


def test_v2_dev_plan_and_oracle_refer_to_exact_existing_lines() -> None:
    plan = _load("review-plan.json")
    oracle = _load("oracle.json")
    assert plan["suite"] == oracle["suite"] == "dev"
    assert plan["world_id"] == oracle["world_id"] == "tidelight-harbor"
    assert oracle["developer_visible"] is True
    candidates = {row["candidate_key"]: row for row in plan["candidate_decisions"]}
    assert len(candidates) == len(oracle["expected_cases"]) == 5
    for candidate in candidates.values():
        assert candidate["source_document"] == "02-character-profiles.md"
        source = _line(candidate["source_document"], candidate["source_line"])
        assert candidate["character_key"] in source
        assert candidate["source_quote"] in source
        if candidate["trait_type"] == "core_personality":
            assert "核心性格" in source or "核心人格" in source
            assert candidate["stability"] == "core"
        else:
            assert candidate["trait_type"] == "preference"
            assert "长期稳定偏好" in source
            assert candidate["stability"] == "stable"
    for case in oracle["expected_cases"]:
        candidate = candidates[case["candidate_key"]]
        assert case["evidence"]["B"] == [{
            "document_name": candidate["source_document"],
            "line": candidate["source_line"],
        }]
        for role in ("B", "C", "G", "X", "forbidden"):
            for ref in case["evidence"][role]:
                expected_file = (
                    "02-character-profiles.md" if role == "B" else
                    "04-draft-event-v1.1.md" if role in {"C", "forbidden"} else
                    "03-published-history-v1.0.md"
                )
                assert ref["document_name"] == expected_file
                source = _line(ref["document_name"], ref["line"])
                if role == "C":
                    assert candidate["character_key"] in source


def test_both_unexplained_violations_explicitly_exclude_secrecy_orders() -> None:
    draft = "04-draft-event-v1.1.md"
    first = _line(draft, 3)
    second = _line(draft, 4)
    for source in (first, second):
        assert "没有封港令" in source
        assert "没有救援保密任务" in source
        assert "桑衍" in source and "俞笙" in source
    assert [ref["line"] for ref in _cases()[
        "dev_repeated_undisclosed_route_risk"
    ]["evidence"]["C"]] == [3, 4]


def test_other_actor_forbidden_anchor_contains_only_the_other_actors_action() -> None:
    draft = "04-draft-event-v1.1.md"
    case = _cases()["dev_other_actor_forces_signature"]
    forbidden = case["evidence"]["forbidden"]
    assert len(forbidden) == 1
    assert forbidden[0]["line"] == 6
    assert forbidden[0]["character_key"] == "陆珩"
    assert forbidden[0]["polarity"] == "positive"
    action = _line(draft, 6)
    assert "叶箫" in action and "签下邱绫的姓名" in action
    assert "陆珩" not in action
    assert "陆珩" in _line(draft, 5)
    assert "陆珩" in _line(draft, 7)


def test_shifted_draft_anchors_remain_on_the_intended_scenes() -> None:
    cases = _cases()
    assert [ref["line"] for ref in cases[
        "dev_published_public_leadership_growth"
    ]["evidence"]["C"]] == [10, 11]
    assert [ref["line"] for ref in cases[
        "dev_published_medical_exception"
    ]["evidence"]["C"]] == [13]
    assert [ref["line"] for ref in cases[
        "dev_staged_villain_confession"
    ]["evidence"]["forbidden"]] == [14]
    assert "许箬" in _line("04-draft-event-v1.1.md", 10)
    assert "许箬" in _line("04-draft-event-v1.1.md", 11)
    assert "荀木" in _line("04-draft-event-v1.1.md", 13)
    assert "陈姝照着反派戏本念台词" in _line("04-draft-event-v1.1.md", 14)
