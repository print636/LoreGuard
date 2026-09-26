from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.character_draft_actor_review import (
    DraftActorProposal,
    build_draft_actor_clause_index,
    screen_draft_actor_proposal,
)
from app.character_scope_review import ScopeReviewSourceIdentity


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "data" / "character-ooc-return-season-dev-v1" / "05-draft-event-v1.2.md"
)


def _source(content: str) -> ScopeReviewSourceIdentity:
    return ScopeReviewSourceIdentity(
        run_input_id="frozen-run-1",
        document_id="draft-1",
        document_version=2,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _review(
    content: str,
    *,
    line_start: int,
    line_end: int | None = None,
    target: str,
    anchor: str,
    character: str,
    statement: str,
    anchor_kind: str = "same_line_corroboration",
    verified: frozenset[str] = frozenset(),
    evidence: str | None = None,
):
    line_end = line_start if line_end is None else line_end
    source = _source(content)
    index = build_draft_actor_clause_index(
        content, source, line_start=line_start, line_end=line_end,
    )
    proposal = DraftActorProposal(
        source_sha256=source.content_sha256,
        line_start=line_start,
        line_end=line_end,
        evidence=evidence if evidence is not None else "\n".join(
            content.splitlines()[line_start - 1:line_end]
        ),
        target_clause_id=target,
        actor_anchor_id=anchor,
        anchor_kind=anchor_kind,
        character=character,
        statement=statement,
    )
    return screen_draft_actor_proposal(
        index, proposal, frozen_content=content, expected_source=source,
        verified_prior_anchor_ids=verified,
    )


def test_d14_is_only_structurally_eligible_for_independent_semantic_review():
    content = FIXTURE.read_text(encoding="utf-8")
    decision = _review(
        content, line_start=20, target="L20:A2", anchor="L20:A5",
        character="柳泛", statement="柳泛又把西侧另一行危险读数直接改成正常值",
    )
    assert decision.eligible_for_semantic_review
    assert decision.reason == "eligible_for_semantic_review"


def test_d06_spliced_voice_cannot_supply_actor():
    content = FIXTURE.read_text(encoding="utf-8")
    decision = _review(
        content, line_start=10, target="L10:A1", anchor="L10:A2",
        character="叶簪", statement="叶簪发布无居籍者不配登船的命令",
    )
    assert decision.reason == "unsafe_context"


def test_d15_apprentice_action_cannot_be_relabelled_as_lufan():
    content = FIXTURE.read_text(encoding="utf-8")
    decision = _review(
        content, line_start=21, target="L21:A5", anchor="L21:A3",
        character="柳泛", statement="柳泛把南侧读数偷偷改低",
    )
    assert not decision.eligible_for_semantic_review
    assert decision.reason == "statement_mismatch"


def test_direct_prior_anchor_requires_server_verified_id_and_adjacency():
    content = "林澈亲自把灯台封好。\n她又把钥匙交给船长。"
    kwargs = dict(
        line_start=1, line_end=2, target="L2:A1", anchor="L1:A1",
        character="林澈", statement="林澈又把钥匙交给船长",
        anchor_kind="verified_prior_named_anchor",
    )
    assert _review(content, **kwargs).reason == "anchor_not_structurally_supported"
    assert _review(content, **kwargs, verified=frozenset({"L1:A1"})).reason == (
        "eligible_for_semantic_review"
    )


@pytest.mark.parametrize("anchor", [
    "林澈的学徒把灯台封好。",
    "林澈亲自让周尧把灯台封好。",
])
def test_possessor_and_delegator_are_not_direct_prior_actor_anchors(anchor):
    content = anchor + "\n她又把钥匙交给船长。"
    decision = _review(
        content, line_start=1, line_end=2, target="L2:A1", anchor="L1:A1",
        character="林澈", statement="林澈又把钥匙交给船长",
        anchor_kind="verified_prior_named_anchor", verified=frozenset({"L1:A1"}),
    )
    assert decision.reason == "anchor_not_structurally_supported"


@pytest.mark.parametrize("content,expected", [
    ("她把原稿改成红色；记录显示本次操作由林澈本人完成。", "eligible_for_semantic_review"),
    ("她和周尧一起把原稿改成红色；记录显示本次操作由林澈本人完成。", "unsafe_context"),
    ("如果计划有变，她把原稿改成红色；记录显示本次操作由林澈本人完成。", "unsafe_context"),
    ("她让周尧把原稿改成红色；记录显示本次操作由林澈本人完成。", "unsafe_context"),
    ("周尧把封面涂蓝。她把原稿改成红色；记录显示本次操作由林澈本人完成。", "unsafe_context"),
    ("她把原稿改成红色，周尧把同页封面涂蓝；记录显示本次操作由林澈本人完成。", "unsafe_context"),
    ("她把原稿改成红色；记录显示本次操作不是林澈本人完成。", "anchor_not_structurally_supported"),
])
def test_screen_rejects_unsafe_grammatical_paths(content, expected):
    source = _source(content)
    index = build_draft_actor_clause_index(content, source, line_start=1, line_end=1)
    target = next(clause for clause in index.lines[0].clauses if clause.text.startswith("她"))
    anchor = index.lines[0].clauses[-1]
    decision = _review(
        content, line_start=1, target=target.support_id, anchor=anchor.support_id,
        character="林澈", statement="林澈" + target.text[1:],
    )
    assert decision.reason == expected


def test_exact_quote_and_frozen_source_are_required():
    content = "她把原稿改成红色；记录显示本次操作由林澈本人完成。"
    decision = _review(
        content, line_start=1, target="L1:A1", anchor="L1:A2",
        character="林澈", statement="林澈把原稿改成红色",
        evidence=content.replace("红色", "蓝色"),
    )
    assert decision.reason == "evidence_mismatch"
    assert _review(
        content, line_start=1, target="L1:A1", anchor="L1:A2",
        character="林澈", statement="林澈把原稿改成蓝色",
    ).reason == "statement_mismatch"
    source = _source(content)
    index = build_draft_actor_clause_index(content, source, line_start=1, line_end=1)
    proposal = DraftActorProposal(
        source_sha256=source.content_sha256, line_start=1, line_end=1,
        evidence=content, target_clause_id="L1:A1", actor_anchor_id="L1:A2",
        anchor_kind="same_line_corroboration", character="林澈",
        statement="林澈把原稿改成红色",
    )
    changed = content.replace("红色", "蓝色")
    assert screen_draft_actor_proposal(
        index, proposal, frozen_content=changed, expected_source=source,
    ).reason == "source_mismatch"


def test_blank_paragraph_cannot_be_used_as_adjacent_actor_anchor():
    content = "林澈亲自把灯台封好。\n\n她又把钥匙交给船长。"
    with pytest.raises(ValueError, match="draft_actor_window_invalid"):
        build_draft_actor_clause_index(
            content, _source(content), line_start=1, line_end=3,
        )
