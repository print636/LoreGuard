from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pytest

from app.character_draft_actor_review import (
    DRAFT_ACTOR_REVIEW_BATCH_SCHEMA_V1,
    DRAFT_ACTOR_REVIEW_SCHEMA_V1,
    DraftActorProposal,
    build_draft_actor_clause_index,
    build_draft_actor_review_request,
    draft_actor_review_batch_digest,
    draft_actor_review_request_digest,
    required_draft_actor_review_basis_ids,
)
from app.character_draft_actor_review_provider import (
    DRAFT_ACTOR_REVIEW_SYSTEM_PROMPT,
    DRAFT_ACTOR_REVIEW_USER_PREFIX,
    DraftActorReviewBatchEntry,
    build_draft_actor_review_prompts,
    run_draft_actor_review,
)
from app.character_scope_review import ScopeReviewSourceIdentity
from app.provider import ProviderError, ProviderRetryExhausted


@dataclass
class Result:
    text: str
    prompt_tokens: int = 10
    completion_tokens: int = 10


class Provider:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def complete(self, system: str, user: str):
        self.calls.append((system, user))
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def _entry(
    character: str, suffix: str, *, successor: str | None = None
) -> DraftActorReviewBatchEntry:
    first = (
        f"她把原稿改成红色；记录显示本次操作由{character}本人完成。"
    )
    content = first if successor is None else f"{first}\n{successor}"
    source = ScopeReviewSourceIdentity(
        run_input_id=f"run-{suffix}",
        document_id=f"draft-{suffix}",
        document_version=1,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    index = build_draft_actor_clause_index(
        content, source, line_start=1, line_end=1
    )
    proposal = DraftActorProposal(
        source_sha256=source.content_sha256,
        line_start=1,
        line_end=1,
        evidence=first,
        target_clause_id="L1:A1",
        actor_anchor_id="L1:A2",
        anchor_kind="same_line_corroboration",
        character=character,
        statement=f"{character}把原稿改成红色",
    )
    request = build_draft_actor_review_request(
        index, (proposal,), frozen_content=content, expected_source=source
    )
    return DraftActorReviewBatchEntry(request, source, content)


def _item(request, **updates):
    item = {
        "proposal_id": request.proposals[0].proposal_id,
        "verdict": "supported",
        "actor": "proposed",
        "actuality": "asserted",
        "statement_relation": "supported",
        "correction_relation": "none",
        "basis_ids": list(required_draft_actor_review_basis_ids(request)),
    }
    item.update(updates)
    return item


def _response(entries, items=None):
    requests = tuple(entry.request for entry in entries)
    if items is None:
        items = tuple(_item(request) for request in requests)
    return json.dumps(
        {
            "schema_version": DRAFT_ACTOR_REVIEW_BATCH_SCHEMA_V1,
            "batch_digest": draft_actor_review_batch_digest(requests),
            "responses": [
                {
                    "schema_version": DRAFT_ACTOR_REVIEW_SCHEMA_V1,
                    "request_digest": draft_actor_review_request_digest(request),
                    "items": [item],
                }
                for request, item in zip(requests, items)
            ],
        },
        ensure_ascii=False,
    )


def _run(entries, provider, **updates):
    values = {
        "token_budget": 100_000,
        "completion_reserve": 512,
        "timeout_seconds": 5,
        "remaining_deadline_seconds": 20,
        "max_response_bytes": 32_768,
        "max_attempts": 1,
    }
    values.update(updates)
    return run_draft_actor_review(
        entries, provider=provider, **values
    )


def test_prompt_treats_anchors_as_nonconclusive_and_covers_nonactual_scenes():
    entry = _entry("林澈", "one")
    system, user = build_draft_actor_review_prompts((entry.request,))
    assert system == DRAFT_ACTOR_REVIEW_SYSTEM_PROMPT
    assert all(word in system for word in ("梦境", "幻觉", "想象", "排练", "演练"))
    assert "不证明人物归属" in system
    assert "不得因为锚点存在、路径齐全" in system
    payload = json.loads(user.removeprefix(DRAFT_ACTOR_REVIEW_USER_PREFIX))
    assert "protocol" not in payload
    assert hashlib.sha256(system.encode("utf-8")).hexdigest() == (
        "c25d43ae2d0b6d32136ad5c4ae017513624a5ab79b1482788e7d41e0eeb5d1e8"
    )
    assert hashlib.sha256(user.encode("utf-8")).hexdigest() == (
        "f807ccd08e0f4096ad8233e547db8bba36fecdb67b40c4a91595263cd6f6024a"
    )
    assert payload["batch_digest"] == draft_actor_review_batch_digest(
        (entry.request,)
    )
    assert payload["windows"][0]["basis_path_hints"][0][
        "required_basis_ids"
    ] == list(required_draft_actor_review_basis_ids(entry.request))


def test_one_provider_call_reviews_multiple_windows():
    entries = (_entry("林澈", "one"), _entry("周砚", "two"))
    provider = Provider(Result(_response(entries)))
    run = _run(entries, provider)
    assert run.attempted_calls == 1
    assert len(provider.calls) == 1
    assert [
        decision.verdict
        for evaluation in run.evaluation.evaluations
        for decision in evaluation.decisions
    ] == ["supported", "supported"]
    assert run.failure_reason is None


def test_provider_all_supported_cannot_bypass_frozen_source_veto():
    entries = (
        _entry(
            "林澈",
            "dream-veto",
            successor="下一幕揭示，上一行只是林澈梦境中的想象。",
        ),
        _entry(
            "周砚",
            "actor-veto",
            successor="更正：上一行动作实际由许棠完成。",
        ),
        _entry(
            "岑野",
            "benign-successor",
            successor="随后，许棠谈起自己追逐多年的梦想。",
        ),
    )
    run = _run(entries, Provider(Result(_response(entries))))
    decisions = [
        evaluation.decisions[0] for evaluation in run.evaluation.evaluations
    ]
    assert [(item.verdict, item.reason) for item in decisions] == [
        ("uncertain", "source_context_veto"),
        ("uncertain", "source_context_veto"),
        ("supported", "supported"),
    ]
    assert decisions[0].basis_ids == decisions[1].basis_ids == ()
    assert run.failure_reason is None


def test_mixed_decisions_require_per_decision_coverage_debt_even_without_failure_reason():
    entries = (_entry("林澈", "one"), _entry("周砚", "two"))
    items = (
        _item(entries[0].request),
        _item(
            entries[1].request,
            verdict="uncertain",
            actor="ambiguous",
            actuality="ambiguous",
            statement_relation="ambiguous",
            correction_relation="ambiguous",
            basis_ids=[],
        ),
    )
    run = _run(entries, Provider(Result(_response(entries, items))))
    assert run.failure_reason is None
    assert [
        decision.verdict
        for evaluation in run.evaluation.evaluations
        for decision in evaluation.decisions
    ] == ["supported", "uncertain"]


def test_source_mismatch_and_token_budget_skip_provider_call():
    entry = _entry("林澈", "one")
    changed = DraftActorReviewBatchEntry(
        entry.request,
        entry.expected_source,
        entry.frozen_content.replace("红色", "蓝色"),
    )
    provider = Provider(Result("unused"))
    mismatch = _run((changed,), provider)
    assert (mismatch.failure_reason, mismatch.attempted_calls) == (
        "source_mismatch", 0
    )
    assert provider.calls == []

    budget = _run((entry,), provider, token_budget=0)
    assert (budget.failure_reason, budget.attempted_calls) == ("token_budget", 0)
    assert provider.calls == []


@pytest.mark.parametrize(
    ("error", "reason"),
    (
        (ProviderError("safe", category="read_timeout"), "provider_timeout"),
        (
            ProviderRetryExhausted(
                "safe", category="rate_limit", http_status=429
            ),
            "provider_rate_limit",
        ),
    ),
)
def test_provider_timeout_and_rate_limit_fail_closed_once(error, reason):
    entry = _entry("林澈", "one")
    provider = Provider(error)
    run = _run((entry,), provider)
    assert (run.failure_reason, run.attempted_calls, len(provider.calls)) == (
        reason, 1, 1
    )
    assert run.evaluation.evaluations[0].decisions[0].verdict == "uncertain"


def test_invalid_response_is_not_repaired_or_persisted():
    entry = _entry("林澈", "one")
    raw = "private-invalid-response"
    run = _run((entry,), Provider(Result(raw)))
    assert (run.failure_reason, run.attempted_calls) == ("response_invalid", 1)
    assert raw not in repr(run)
    assert run.evaluation.evaluations[0].decisions[0].reason == "response_invalid"


def test_max_attempts_must_remain_one():
    entry = _entry("林澈", "one")
    with pytest.raises(ValueError, match="attempt limit"):
        _run((entry,), Provider(Result(_response((entry,)))), max_attempts=2)
