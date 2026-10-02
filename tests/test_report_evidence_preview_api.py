from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app import main
from app.auth import AuthContext, get_auth_context
from app.character_consistency_stage import (
    PROVISIONAL_DRAFT_CLUES_PAYLOAD_KEY,
    ProvisionalDraftClue,
    _provisional_clue_id,
)
from app.db import (
    AnalysisDiagnosticRow,
    AnalysisRunCharacterTraitInputRow,
    AnalysisRunInputContextRow,
    AnalysisRunInputRow,
    AnalysisRunRow,
    Base,
    CharacterTraitCandidateRow,
    CharacterTraitReviewRow,
    DocumentContextRow,
    DocumentRow,
    FeedbackRow,
    IssueRow,
    ProjectRow,
    WorkspaceRow,
)
from app.narrative_context import payload_sha256


def _hash(text):
    return sha256(text.encode("utf-8")).hexdigest()


def _span(source, start, end=None):
    end = start if end is None else end
    return {
        "document_id": source.document_id,
        "document_name": source.document_name,
        "line_start": start, "line_end": end,
        "text": "\n".join(source.content.splitlines()[start - 1:end]).strip(),
    }


def _issue(run, spans, *, report_class="formal", metadata=None):
    return IssueRow(
        id=str(uuid4()), run_id=run.id, report_class=report_class,
        category="character_drift", severity="medium", confidence=0.6,
        title="保存的报告条目", explanation="阅读来源绑定合同。", suggestion="核对原文。",
        evidence=spans,
        extra=metadata or {
            "final_outcome": "conflict" if report_class == "formal" else "needs_confirmation",
            "review_reason": "reading_contract_fixture",
        },
    )


def _forbidden_runtime_call(*_args, **_kwargs):
    raise AssertionError("Evidence reading must not call providers or dispatch analysis")


@pytest.fixture()
def evidence_api(tmp_path, monkeypatch):
    # Every endpoint query belongs to this fixture's temporary DB. No lifespan,
    # developer .env, real loreguard.db, evaluation cases or provider is used.
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'report-evidence-preview.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    with sessions() as db:
        workspace = WorkspaceRow(id=str(uuid4()), name="阅读工作区")
        foreign_workspace = WorkspaceRow(id=str(uuid4()), name="另一工作区")
        db.add_all([workspace, foreign_workspace])
        db.flush()
        project = ProjectRow(workspace_id=workspace.id, name="阅读项目")
        other_project = ProjectRow(workspace_id=workspace.id, name="同工作区另一项目")
        foreign_project = ProjectRow(workspace_id=foreign_workspace.id, name="另一工作区项目")
        db.add_all([project, other_project, foreign_project])
        db.flush()
        run = AnalysisRunRow(project_id=project.id, status="completed")
        old_run = AnalysisRunRow(project_id=project.id, status="completed")
        other_run = AnalysisRunRow(project_id=project.id, status="completed")
        other_project_run = AnalysisRunRow(project_id=other_project.id, status="completed")
        foreign_run = AnalysisRunRow(project_id=foreign_project.id, status="completed")
        db.add_all([run, old_run, other_run, other_project_run, foreign_run])
        db.flush()
        body_lines = [f"第{number}行。" for number in range(1, 146)]
        body_lines[13:16] = ["  第十四行 😀 [a.*]。  ", "", "第十六行，保留 Unicode。"]
        body = "\r\n".join(body_lines) + "\r\n"
        document = DocumentRow(project_id=project.id, name="draft.md", content=body, version=1)
        db.add(document)
        db.flush()
        db.add(DocumentContextRow(document_id=document.id, document_role="chapter", story_scope="main"))
        frozen = AnalysisRunInputRow(
            run_id=run.id, document_id=document.id, document_name=document.name,
            document_version=1, content=body, content_sha256=_hash(body), ordinal=0,
        )
        shared_id = str(uuid4())
        original = AnalysisRunInputRow(
            run_id=old_run.id, document_id=shared_id, document_name="baseline.md",
            document_version=1, content="旧版本前文。\n冻结基准证据。\n旧版本末行。",
            content_sha256=_hash("旧版本前文。\n冻结基准证据。\n旧版本末行。"), ordinal=0,
        )
        replacement = AnalysisRunInputRow(
            run_id=run.id, document_id=shared_id, document_name="baseline.md",
            document_version=2, content="新版本前文。\n冻结基准证据。\n新版本末行。",
            content_sha256=_hash("新版本前文。\n冻结基准证据。\n新版本末行。"), ordinal=1,
        )
        foreign = AnalysisRunInputRow(
            run_id=foreign_run.id, document_id=str(uuid4()), document_name="foreign.md",
            document_version=1, content="另一工作区正文。", content_sha256=_hash("另一工作区正文。"), ordinal=0,
        )
        other_project_source = AnalysisRunInputRow(
            run_id=other_project_run.id, document_id=str(uuid4()), document_name="other.md",
            document_version=1, content="另一项目正文。", content_sha256=_hash("另一项目正文。"), ordinal=0,
        )
        db.add_all([frozen, original, replacement, foreign, other_project_source])
        db.flush()
        db.add(AnalysisRunInputContextRow(input_id=frozen.id, document_role="chapter", story_scope="main"))
        formal = _issue(run, [_span(frozen, 14, 16)])
        review = _issue(run, [_span(frozen, 14), _span(frozen, 16)], report_class="review_clue")
        other_issue = _issue(other_run, [_span(frozen, 14)])
        foreign_issue = _issue(foreign_run, [_span(foreign, 1)])
        db.add_all([formal, review, other_issue, foreign_issue])

        evidence = [{
            **_span(original, 2), "input_id": original.id,
            "document_version": original.document_version,
            "content_sha256": original.content_sha256,
        }]
        candidate = CharacterTraitCandidateRow(
            project_id=project.id, source_run_id=old_run.id,
            character_key="阅读角色", character_display_name="阅读角色",
            trait_type="core_personality", trait_key="reading_fixture", value="冻结资料",
            polarity="unclear", stability="stable", contexts=[], origin="explicit_setting",
            authority_tier="formal_record", confidence=0.6,
            scope_payload={}, scope_sha256=payload_sha256({}),
            evidence=evidence, evidence_sha256=payload_sha256(evidence),
            support_binding_mode="legacy_v1", candidate_fingerprint="1" * 64,
            generator_version="reading_contract_v1", review_state="confirmed", lock_version=1,
        )
        db.add(candidate)
        db.flush()
        confirmation = CharacterTraitReviewRow(
            project_id=project.id, candidate_id=candidate.id, decision="confirm", expected_lock_version=0,
        )
        db.add(confirmation)
        db.flush()
        trait_payload = {
            "candidate_id": candidate.id, "confirmation_review_id": confirmation.id,
            "candidate_lock_version": 1, "evidence": evidence,
            "evidence_sha256": payload_sha256(evidence),
        }
        trait = AnalysisRunCharacterTraitInputRow(
            run_id=run.id, project_id=project.id, candidate_id=candidate.id,
            confirmation_review_id=confirmation.id, candidate_lock_version=1, ordinal=0,
            payload=trait_payload, payload_sha256=payload_sha256(trait_payload),
        )
        metadata = {
            "confirmed_candidate_id": candidate.id,
            "final_outcome": "needs_confirmation", "review_reason": "reading_contract_fixture",
            "evidence_binding": "server_evidence_pair_v1",
        }
        trait_review = _issue(
            run, [_span(original, 2), _span(frozen, 14)],
            report_class="review_clue", metadata=metadata,
        )
        trait_issue = _issue(run, [_span(original, 2), _span(frozen, 14)], metadata={
            **metadata, "final_outcome": "conflict",
        })
        db.add_all([trait, trait_review, trait_issue])

        provisional = ProvisionalDraftClue(
            id="pc_" + "0" * 32, input_id=frozen.id, content_sha256=frozen.content_sha256,
            document_id=frozen.document_id, document_version=1, document_name=frozen.document_name,
            line_start=14, line_end=14, evidence=_span(frozen, 14)["text"],
            character="阅读角色", dimension="core_personality",
            proposed_statement="待验证的模型提案。", reason="partial_model_package",
        )
        provisional = provisional.model_copy(update={"id": _provisional_clue_id(provisional)})
        diagnostic = AnalysisDiagnosticRow(run_id=run.id, payload={
            PROVISIONAL_DRAFT_CLUES_PAYLOAD_KEY: {
                "items": [provisional.model_dump(mode="json")], "truncated": False,
            },
        })
        db.add(diagnostic)
        db.commit()

    monkeypatch.setattr(main, "SessionLocal", sessions)
    monkeypatch.setitem(main.app.dependency_overrides, get_auth_context, lambda: AuthContext(
        user_id=str(uuid4()), workspace_id=workspace.id, role="owner", anonymous=False,
    ))
    for name in (
        "OpenAICompatibleProvider", "account_provider_runtime", "dispatch_analysis",
        "infer_narrative_context", "_narrative_context_inference_provider", "execute_analysis",
    ):
        monkeypatch.setattr(main, name, _forbidden_runtime_call)
    state = SimpleNamespace(**locals(), statements=[], reading=False)

    @event.listens_for(engine, "before_cursor_execute")
    def only_select(_connection, _cursor, statement, _params, _context, _many):
        if state.reading:
            state.statements.append(statement)
            assert statement.lstrip().upper().startswith("SELECT"), statement

    state.client = TestClient(main.app)
    try:
        yield state
    finally:
        state.client.close()
        engine.dispose()


@contextmanager
def _read_only(state):
    state.reading = True
    try:
        yield
    finally:
        state.reading = False


def _get(state, *, kind="issue", item=None, run=None, **parameters):
    item = state.formal if item is None else item
    run = state.run if run is None else run
    item_id = item.id if hasattr(item, "id") else item
    run_id = run.id if hasattr(run, "id") else run
    with _read_only(state):
        return state.client.get(
            f"/api/v1/analysis-runs/{run_id}/evidence-preview",
            params={"kind": kind, "item_id": item_id, **parameters},
        )


def test_preview_reads_old_run_body_after_live_document_and_context_update(evidence_api):
    state = evidence_api
    before = _get(state)
    assert before.status_code == 200, before.text
    with state.sessions() as db:
        document = db.get(DocumentRow, state.document.id)
        document.content, document.version, document.active = "修改后的当前正文。", 2, False
        context = db.get(DocumentContextRow, state.document.id)
        context.document_role, context.story_scope = "setting", "changed_scope"
        db.commit()
    after = _get(state)
    assert after.json() == before.json()
    assert after.headers["Cache-Control"] == "no-store"
    payload = after.json()
    assert set(payload) == {"run", "reference", "source", "lines", "page", "message"}
    assert payload["source"] == {
        "availability": "full_context", "provenance": "run_input", "source_run_id": state.run.id,
        "document_version": 1, "content_sha256": state.frozen.content_sha256,
        "char_count": len(state.frozen.content), "line_count": 145,
    }
    assert payload["reference"] == {
        "kind": "issue", "item_id": state.formal.id, "evidence_index": 0,
        **state.formal.evidence[0],
    }
    assert payload["page"] == {"offset": 1, "limit": 120, "total": 145, "has_more": True}
    assert [row["line_number"] for row in payload["lines"] if row["is_evidence"]] == [14, 15, 16]
    assert payload["lines"][12]["text"] == "  第十四行 😀 [a.*]。  "
    assert payload["lines"][13]["text"] == ""
    assert payload["message"] is None
    assert state.statements
    assert not any("FROM documents" in query for query in state.statements)


@pytest.mark.parametrize("offset,limit,numbers,more", [
    (13, 3, [14, 15, 16], True), (144, 200, [145], False),
    (145, 120, [], False), (1000, 120, [], False),
])
def test_preview_preserves_source_line_numbers_and_page_bounds(evidence_api, offset, limit, numbers, more):
    response = _get(evidence_api, offset=offset, limit=limit)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert [row["line_number"] for row in payload["lines"]] == numbers
    assert payload["page"] == {"offset": offset, "limit": limit, "total": 145, "has_more": more}


def test_report_categories_keep_their_existing_status_and_feedback_boundary(evidence_api):
    state = evidence_api
    with _read_only(state):
        issues = state.client.get(f"/api/v1/analysis-runs/{state.run.id}/issues").json()
        clues = state.client.get(f"/api/v1/analysis-runs/{state.run.id}/review-clues").json()["items"]
        provisional = state.client.get(f"/api/v1/analysis-runs/{state.run.id}/provisional-clues").json()["items"]
    issue_before = next(item for item in issues if item["id"] == state.formal.id)
    clue_before = next(item for item in clues if item["id"] == state.review.id)
    assert issue_before["report_class"] == "formal"
    assert clue_before["report_class"] == "review_clue"
    for kind, item in [("issue", state.formal), ("review_clue", state.review), ("provisional_clue", state.provisional)]:
        response = _get(state, kind=kind, item=item)
        assert response.status_code == 200, response.text
        assert response.json()["reference"]["kind"] == kind
    assert _get(state, kind="provisional_clue", item=state.provisional).json()["reference"]["text"] == provisional[0]["evidence"]
    with state.sessions() as db:
        assert main._serialize_issue(db.get(IssueRow, state.formal.id)) == issue_before
        assert main._serialize_issue(db.get(IssueRow, state.review.id)) == clue_before
        assert db.scalars(select(FeedbackRow)).all() == []
        assert db.get(CharacterTraitCandidateRow, state.candidate.id).review_state == "confirmed"
        assert db.get(AnalysisRunRow, state.run.id).status == "completed"


@pytest.mark.parametrize("kind,item_name,index", [
    ("issue", "trait_issue", 0), ("review_clue", "trait_review", 0),
])
def test_confirmed_trait_uses_original_input_even_when_current_version_has_same_excerpt(evidence_api, kind, item_name, index):
    state = evidence_api
    response = _get(state, kind=kind, item=getattr(state, item_name), evidence_index=index)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["source"]["provenance"] == "confirmed_trait_source"
    assert payload["source"]["source_run_id"] == state.old_run.id
    assert payload["source"]["document_version"] == 1
    assert payload["source"]["content_sha256"] == state.original.content_sha256
    assert [row["text"] for row in payload["lines"]] == state.original.content.splitlines()
    assert "新版本前文" not in response.text
    current = _get(state, kind=kind, item=getattr(state, item_name), evidence_index=1)
    assert current.json()["source"]["provenance"] == "run_input"
    assert current.json()["source"]["source_run_id"] == state.run.id


@pytest.mark.parametrize("role_case", ["missing_candidate", "missing_exact_binding", "current_same_excerpt"])
def test_explicit_source_roles_cannot_borrow_a_different_frozen_body(evidence_api, role_case):
    state = evidence_api
    with state.sessions() as db:
        issue = db.get(IssueRow, state.trait_issue.id)
        if role_case == "missing_candidate":
            issue.extra = {**issue.extra, "confirmed_candidate_id": str(uuid4())}
        elif role_case == "missing_exact_binding":
            trait = db.get(AnalysisRunCharacterTraitInputRow, state.trait.id)
            evidence = [{**trait.payload["evidence"][0], "text": "不同冻结摘录。"}]
            trait.payload = {**trait.payload, "evidence": evidence, "evidence_sha256": payload_sha256(evidence)}
            trait.payload_sha256 = payload_sha256(trait.payload)
        else:
            # C has the same five-field excerpt as frozen B, but C must resolve
            # against the current run's input, including its current version.
            issue.evidence = [issue.evidence[1], _span(state.replacement, 2)]
        db.commit()
    response = _get(state, item=state.trait_issue, evidence_index=1 if role_case == "current_same_excerpt" else 0)
    if role_case == "current_same_excerpt":
        assert response.status_code == 200, response.text
        assert response.json()["source"]["provenance"] == "run_input"
        assert response.json()["source"]["document_version"] == 2
        assert response.json()["lines"][0]["text"] == "新版本前文。"
    else:
        assert response.status_code == 409, response.text
        assert response.json() == {"detail": "报告证据原文快照无法核验"}


@pytest.mark.parametrize("kind,item_name", [("issue", "trait_issue"), ("review_clue", "trait_review")])
def test_verified_trait_missing_full_source_returns_only_frozen_excerpt(evidence_api, kind, item_name):
    state = evidence_api
    with state.sessions() as db:
        db.delete(db.get(AnalysisRunInputRow, state.original.id))
        db.commit()
    response = _get(state, kind=kind, item=getattr(state, item_name))
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["source"] == {
        "availability": "excerpt_only", "provenance": "confirmed_trait_source",
        "source_run_id": None, "document_version": 1,
        "content_sha256": state.original.content_sha256, "char_count": None, "line_count": None,
    }
    assert payload["reference"]["text"] == "冻结基准证据。"
    assert payload["lines"] == [] and payload["page"] is None
    assert payload["message"] == "完整原文快照已缺失，仅可查看已冻结的证据摘录。"
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("case", [
    "unknown_run", "foreign_run", "unknown_item", "other_run_item", "foreign_item",
    "wrong_kind", "wrong_review_kind", "unknown_index", "provisional_index",
])
def test_unknown_and_ownership_failures_have_same_private_not_found(evidence_api, case):
    state = evidence_api
    arguments = {
        "unknown_run": {"run": str(uuid4())},
        "foreign_run": {"run": state.foreign_run, "item": state.foreign_issue},
        "unknown_item": {"item": str(uuid4())},
        "other_run_item": {"item": state.other_issue},
        "foreign_item": {"item": state.foreign_issue},
        "wrong_kind": {"kind": "review_clue"},
        "wrong_review_kind": {"kind": "issue", "item": state.review},
        "unknown_index": {"evidence_index": 11},
        "provisional_index": {"kind": "provisional_clue", "item": state.provisional, "evidence_index": 1},
    }[case]
    response = _get(state, **arguments)
    assert response.status_code == 404, response.text
    assert response.json() == {"detail": "报告证据不存在"}
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("changes", [
    {"kind": "other"}, {"item": "x" * 201}, {"item": ""},
    {"evidence_index": -1}, {"evidence_index": 12}, {"evidence_index": "invalid"},
    {"offset": -1}, {"offset": "invalid"}, {"limit": 0}, {"limit": 201},
])
def test_invalid_parameters_are_rejected_before_snapshot_reading(evidence_api, changes):
    response = _get(evidence_api, **changes)
    assert response.status_code == 422, response.text
    assert evidence_api.statements == []


@pytest.mark.parametrize("mutation", ["missing", "body", "hash", "name", "range", "excerpt", "trait_hash", "old_body", "old_version"])
def test_missing_or_corrupt_source_never_exposes_guessed_context(evidence_api, mutation):
    state = evidence_api
    item = state.formal
    with state.sessions() as db:
        source = db.get(AnalysisRunInputRow, state.frozen.id)
        if mutation == "missing":
            db.delete(source)
        elif mutation == "body":
            source.content = "替换的原文。"
        elif mutation == "hash":
            source.content_sha256 = "0" * 64
        elif mutation == "name":
            source.document_name = "renamed.md"
        elif mutation in {"range", "excerpt"}:
            issue = db.get(IssueRow, state.formal.id)
            issue.evidence = [{
                **issue.evidence[0],
                **({"line_end": 1000} if mutation == "range" else {"text": "伪造摘录。"}),
            }]
        elif mutation == "trait_hash":
            db.get(AnalysisRunCharacterTraitInputRow, state.trait.id).payload_sha256 = "0" * 64
            item = state.trait_issue
        elif mutation == "old_body":
            db.get(AnalysisRunInputRow, state.original.id).content = "损坏的旧原文。"
            item = state.trait_issue
        elif mutation == "old_version":
            db.get(AnalysisRunInputRow, state.original.id).document_version = 2
            item = state.trait_issue
        db.commit()
    response = _get(state, item=item)
    assert response.status_code == 409, response.text
    assert response.json()["detail"] in {"报告证据原文快照缺失", "报告证据原文快照无法核验"}
    assert response.headers["Cache-Control"] == "no-store"
    assert "替换的原文" not in response.text and "损坏的旧原文" not in response.text


@pytest.mark.parametrize("source_name", ["foreign", "other_project_source"])
def test_trait_binding_cannot_follow_input_into_another_project_or_workspace(evidence_api, source_name):
    state = evidence_api
    source = getattr(state, source_name)
    with state.sessions() as db:
        trait = db.get(AnalysisRunCharacterTraitInputRow, state.trait.id)
        evidence = [{
            **_span(source, 1), "input_id": source.id,
            "document_version": source.document_version, "content_sha256": source.content_sha256,
        }]
        trait.payload = {**trait.payload, "evidence": evidence, "evidence_sha256": payload_sha256(evidence)}
        trait.payload_sha256 = payload_sha256(trait.payload)
        issue = db.get(IssueRow, state.trait_issue.id)
        issue.evidence = [_span(source, 1), issue.evidence[1]]
        db.commit()
    response = _get(state, item=state.trait_issue)
    assert response.status_code == 404, response.text
    assert response.json() == {"detail": "报告证据不存在"}
    assert response.headers["Cache-Control"] == "no-store"
    assert "另一工作区正文" not in response.text and "另一项目正文" not in response.text


def test_arbitrary_document_and_text_parameters_cannot_replace_saved_reference(evidence_api):
    state = evidence_api
    response = _get(
        state, document_id=state.foreign.document_id, text="伪造原文。", line_start=1, line_end=1,
    )
    assert response.status_code == 200, response.text
    assert response.json()["reference"]["document_id"] == state.frozen.document_id
    assert response.json()["reference"]["text"] == state.formal.evidence[0]["text"]
    assert "伪造原文" not in response.text


def test_hidden_review_clue_and_provisional_tampering_do_not_gain_read_access(evidence_api):
    state = evidence_api
    with state.sessions() as db:
        review = db.get(IssueRow, state.review.id)
        review.extra = {**review.extra, "final_outcome": "conflict"}
        diagnostic = db.get(AnalysisDiagnosticRow, state.run.id)
        private = diagnostic.payload[PROVISIONAL_DRAFT_CLUES_PAYLOAD_KEY]
        diagnostic.payload = {PROVISIONAL_DRAFT_CLUES_PAYLOAD_KEY: {
            **private, "items": [{**private["items"][0], "input_id": state.foreign.id}],
        }}
        db.commit()
    assert _get(state, kind="review_clue", item=state.review).status_code == 404
    response = _get(state, kind="provisional_clue", item=state.provisional)
    assert response.status_code == 409, response.text
    assert response.headers["Cache-Control"] == "no-store"
    assert "另一工作区正文" not in response.text
