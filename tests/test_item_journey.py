"""Persisted synthetic extraction -> authorized read-only journey and reader.

These tests verify projection contracts, not live model reasoning quality.
"""
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.config import NARRATIVE_TIME_SCOPE_POLICY_V1, SEMANTIC_STATE_POLICY_V1
from app.db import (AnalysisDiagnosticRow, AnalysisRecordRow, AnalysisRunInputContextRow,
                    AnalysisRunInputRow, AnalysisRunRow, IssueRow)
from app.item_journey import _key
from app.narrative_time_scope import TIME_SCOPE_REFS_KEY
from app.narrative_time_scope_storage import time_scope_storage_payload
from app.semantic_state import SEMANTIC_STATE_REFS_KEY, digest
from app.semantic_state_storage import semantic_state_storage_payload
from tests.test_narrative_time_scope import run_chain
from tests.test_report_evidence_preview_api import evidence_api, _read_only


def stored_chain(api, *, mode=None, prefix_time=None):
    def mutate(raw):
        scope = raw[0][-1]
        if mode == "probable":
            scope["certainty"] = "probable"
        elif mode == "omitted":
            scope["omitted_count"] = 1
        elif mode == "missing_member":
            scope["members"] = scope["members"][:2]
    result, calls, documents = run_chain(mutate=mutate, **({"last_time":prefix_time} if prefix_time else {}))
    assert len(calls) == 1  # MockTransport; query itself must make zero calls.
    with api.sessions() as db:
        run = AnalysisRunRow(project_id=api.project.id, status="completed", batch_coverage={
            "semantic_state_policy": SEMANTIC_STATE_POLICY_V1,
            "narrative_time_scope_policy": NARRATIVE_TIME_SCOPE_POLICY_V1})
        db.add(run)
        db.flush()
        for ordinal, document in enumerate(documents):
            source = AnalysisRunInputRow(run_id=run.id, document_id=document.id,
                document_name=document.name, document_version=4, content=document.content,
                content_sha256=sha256(document.content.encode()).hexdigest(), ordinal=ordinal)
            db.add(source)
            db.flush()
            db.add(AnalysisRunInputContextRow(input_id=source.id, document_role=document.role, story_scope=document.scope))
        db.flush()
        records = [AnalysisRecordRow(id=str(uuid4()), run_id=run.id, kind=row.kind,
            attrs=dict(row.attrs), evidence=row.evidence.model_dump()) for row in result.directives]
        state = semantic_state_storage_payload(db, run=run, records=records, directives=result.directives)
        scopes = time_scope_storage_payload(db, run=run, records=records, directives=result.directives)
        payload = {SEMANTIC_STATE_REFS_KEY: state, TIME_SCOPE_REFS_KEY: scopes,
                   "model": {"partial_fallback": mode is not None}}
        if mode == "legacy":
            payload = {"model": {"partial_fallback": False}}
        if mode == "scope_missing":
            payload.pop(TIME_SCOPE_REFS_KEY)
        if mode == "proof_damaged":
            payload[SEMANTIC_STATE_REFS_KEY]["receipts"][0]["record_sha256"] = "0" * 64
            payload[SEMANTIC_STATE_REFS_KEY]["payload_sha256"] = digest({k:v for k,v in state.items() if k != "payload_sha256"})
        db.add_all(records)
        db.add(AnalysisDiagnosticRow(run_id=run.id, payload=payload))
        item_rows = [row for row in records if row.kind in {"item", "uses"}]
        issue = IssueRow(run_id=run.id, report_class="formal", category="item_ownership",
            severity="high", confidence=.9, title="合成引用条目", explanation="引用绑定测试",
            suggestion="核对", extra={"item": "铜钥匙"}, evidence=[item_rows[0].evidence])
        db.add(issue)
        db.commit()
        return SimpleNamespace(run=run, records=records, item_rows=item_rows, issue=issue,
                               result=result, key=_key(run.id, "铜钥匙", "main"))


def get(api, chain, **params):
    with _read_only(api):
        response = api.client.get(f"/api/v1/analysis-runs/{chain.run.id}/item-journeys/{chain.key}", params=params)
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "no-store"
    return response.json()


def test_persisted_chain_summary_relations_pagination_and_frozen_record_reader(evidence_api):
    api = evidence_api
    chain = stored_chain(api)
    with _read_only(api):
        listed = api.client.get(f"/api/v1/analysis-runs/{chain.run.id}/item-journeys", params={"q":"铜", "issue_id":chain.issue.id}).json()
    assert listed["items"] == [{"key":chain.key,"name":"铜钥匙","story_scope":"main",
                                "record_count":3,"issue_ids":[chain.issue.id]}]
    view = get(api, chain, limit=1)
    assert view["summary"]["status"] == "ordered_evidence"
    assert "沈澈使用" in view["summary"]["text"]
    assert "此后持续保管" in view["summary"]["text"]
    assert view["summary"]["latest_record_ids"] == [next(r.id for r in chain.item_rows if r.kind == "uses")]
    assert view["events"]["total"] == 3 and view["events"]["has_more"]
    assert len(view["relations"]) == 3 and all(r["evidence_record_ids"] for r in view["relations"])
    assert len(view["relation_evidence"]) == 1
    assert view["coverage"]["unresolved_time_pairs"] == 0
    assert not view["coverage"]["proof_unavailable"]
    all_events = get(api, chain)["events"]["items"]
    assert [(e["transfer_from"], e["holder"]) for e in all_events if e["kind"] == "transfer"] == [("库管", "许舟"), ("许舟", "沈澈")]
    assert all(e["basis"] == "supported_inference" and e["document_version"] == 4 for e in all_events)
    assert not get(api, chain, offset=2, limit=1)["events"]["has_more"]
    for item in [*all_events, *view["relation_evidence"]]:
        with _read_only(api):
            preview = api.client.get(f"/api/v1/analysis-runs/{chain.run.id}/evidence-preview",
                params={"kind":"record", "item_id":item["record_id"]})
        assert preview.status_code == 200, preview.text
        assert preview.json()["source"]["document_version"] == 4
        assert preview.json()["reference"]["text"] == item["evidence"]["text"]
    assert "binding" not in str(view) and "payload_sha256" not in str(view)


@pytest.mark.parametrize("mode", ["legacy", "scope_missing", "proof_damaged", "probable", "omitted", "missing_member"])
def test_missing_or_uncertain_proofs_never_choose_last_text_line(evidence_api, mode):
    chain = stored_chain(evidence_api, mode=mode)
    view = get(evidence_api, chain)
    assert view["summary"]["status"] in {"partial", "unavailable"}
    assert view["summary"]["latest_record_ids"] == []
    assert view["coverage"]["unresolved_time_pairs"] > 0
    assert all(e["source_status"] == "verified" for e in view["events"]["items"])


@pytest.mark.parametrize("damage", ["body", "missing", "record", "scope_members"])
def test_damaged_sources_records_and_endpoint_bindings_degrade_without_mutation(evidence_api, damage):
    api = evidence_api
    chain = stored_chain(api)
    with api.sessions() as db:
        if damage in {"body", "missing"}:
            source = db.scalar(select(AnalysisRunInputRow).where(AnalysisRunInputRow.run_id == chain.run.id,
                                                              AnalysisRunInputRow.document_id == "b"))
            if damage == "body": source.content = "已改变的原文"
            else: db.delete(source)
        elif damage == "record":
            row = db.get(AnalysisRecordRow, chain.item_rows[-1].id)
            row.attrs = {**row.attrs, "user": "另一个人"}
        else:
            diagnostic = db.get(AnalysisDiagnosticRow, chain.run.id)
            payload = deepcopy(diagnostic.payload)
            receipt = payload[TIME_SCOPE_REFS_KEY]["receipts"][0]
            receipt["members"][0]["matched_record_ids"] = [chain.item_rows[-1].id]
            envelope = payload[TIME_SCOPE_REFS_KEY]
            envelope["payload_sha256"] = digest({k:v for k,v in envelope.items() if k != "payload_sha256"})
            diagnostic.payload = payload
        db.commit()
    view = get(api, chain)
    assert not view["summary"]["latest_record_ids"]
    if damage in {"body", "missing"}:
        assert view["coverage"]["source_unavailable"] == 2
        assert sum(e["evidence"] is None for e in view["events"]["items"]) == 2
        with _read_only(api):
            reader = api.client.get(f"/api/v1/analysis-runs/{chain.run.id}/evidence-preview",
                params={"kind":"record","item_id":chain.item_rows[-1].id})
        assert reader.status_code == 409


def test_run_branch_and_issue_isolation_and_explicit_limits(evidence_api, monkeypatch):
    api = evidence_api
    chain = stored_chain(api)
    with api.sessions() as db:
        row = AnalysisRecordRow(run_id=chain.run.id, kind="item", attrs={"item":"铜钥匙", "owner":"别线人物", "story_scope":"other"},
                                evidence=chain.item_rows[0].evidence)
        db.add(row)
        db.commit()
    with _read_only(api):
        listing = api.client.get(f"/api/v1/analysis-runs/{chain.run.id}/item-journeys",params={"limit":1}).json()
        assert listing["page"] == dict(offset=0,limit=1,total=2,has_more=True)
        assert api.client.get(f"/api/v1/analysis-runs/{api.foreign_run.id}/item-journeys").status_code == 404
        assert api.client.get(f"/api/v1/analysis-runs/{api.other_run.id}/item-journeys/{chain.key}").status_code == 404
        assert api.client.get(f"/api/v1/analysis-runs/{chain.run.id}/item-journeys",params={"issue_id":api.foreign_issue.id}).status_code == 404
        assert api.client.get(f"/api/v1/analysis-runs/{api.other_run.id}/evidence-preview",params={"kind":"record","item_id":chain.item_rows[0].id}).status_code == 404
        assert api.client.get(f"/api/v1/analysis-runs/{chain.run.id}/evidence-preview",params={"kind":"record","item_id":chain.item_rows[0].id,"evidence_index":1}).status_code == 404
        assert api.client.get(f"/api/v1/analysis-runs/{chain.run.id}/item-journeys",params={"limit":101}).status_code == 422
    assert get(api, chain)["coverage"]["total_records"] == 3
    monkeypatch.setattr("app.item_journey.MAX_JOURNEY_RECORDS", 2)
    view = get(api, chain)
    assert view["coverage"]["truncated"] and view["coverage"]["processed_records"] == 2
    assert view["events"]["total"] == 3 and not view["summary"]["latest_record_ids"]
    monkeypatch.setattr("app.item_journey.MAX_CATALOG_RECORDS", 1)
    with _read_only(api):
        listing = api.client.get(f"/api/v1/analysis-runs/{chain.run.id}/item-journeys").json()
    assert listing["coverage"] == dict(scanned_records=1,total_records=4,truncated=True)


def add_legacy_relation(api, chain, *, scope="main", times=("09:00", "09:30"), damage=False):
    """A real persisted legacy shape; no invented endpoint certificate."""
    text = (f"另一件物品的取用发生在{times[0]}与{times[1]}，原关系未保存端点证明。"
            if times else "另一条记录提到两次取用的先后，原关系未保存端点证明。")
    with api.sessions() as db:
        source = AnalysisRunInputRow(run_id=chain.run.id, document_id=str(uuid4()), document_name="other.md",
            document_version=1, content=text, content_sha256=sha256(text.encode()).hexdigest(), ordinal=2)
        db.add(source)
        db.flush()
        db.add(AnalysisRunInputContextRow(input_id=source.id, document_role="chapter", story_scope=scope))
        db.add(AnalysisRecordRow(run_id=chain.run.id, kind="narrative_time_relation",
            attrs={"document_role":"chapter", "story_scope":scope, "relation":"before",
                   **({"left_time":times[0], "right_time":times[1]} if times else {})},
            evidence=dict(document_id=source.document_id,document_name=source.document_name,
                          line_start=1,line_end=1,text=text)))
        if damage:
            source.content_sha256 = "0" * 64
        db.commit()


def test_proven_terminal_survives_unknown_earlier_pair_without_clearing_its_debt(evidence_api):
    chain = stored_chain(evidence_api)
    add_legacy_relation(evidence_api, chain)
    view = get(evidence_api, chain)
    terminal = next(row.id for row in chain.item_rows if row.kind == "uses")
    assert view["summary"]["latest_record_ids"] == [terminal]
    assert view["summary"]["status"] == "ordered_evidence"
    assert "1组记录的相互先后仍未明" in view["summary"]["text"]
    assert view["coverage"]["unresolved_time_pairs"] == 1
    assert view["coverage"]["relation_proof_unavailable"] == 1
    assert len(view["relations"]) == 2
    assert all(terminal in {r["left_record_id"],r["right_record_id"]} for r in view["relations"])


@pytest.mark.parametrize("mode", ["different_scope", "other_endpoints", "missing_endpoints", "damaged_scope_source"])
def test_only_potentially_relevant_unverified_relations_block_a_terminal(evidence_api, mode):
    chain = stored_chain(evidence_api)
    add_legacy_relation(evidence_api, chain,
        scope="branch_b" if mode in {"different_scope", "damaged_scope_source"} else "main",
        times=None if mode == "missing_endpoints" else (("08:00", "08:30") if mode == "other_endpoints" else ("09:00", "当天九点半")),
        damage=mode == "damaged_scope_source")
    view = get(evidence_api, chain)
    if mode in {"different_scope", "other_endpoints"}:
        assert view["summary"]["latest_record_ids"] == [next(row.id for row in chain.item_rows if row.kind == "uses")]
        assert view["coverage"]["relation_proof_unavailable"] == 0
        assert view["coverage"]["unresolved_time_pairs"] == 0
    else:
        assert view["summary"]["latest_record_ids"] == []
        assert view["coverage"]["relation_proof_unavailable"] == 1
        assert view["coverage"]["unresolved_time_pairs"] == 3


def test_legacy_clock_cannot_exclude_a_sourced_narrative_time_prefix(evidence_api):
    # The source's 当天10:00 is a legitimate endpoint of an ordered scope.
    # An uncertified legacy 10:00 may refer to it; spelling cannot clear debt.
    chain = stored_chain(evidence_api, prefix_time="当天10:00")
    add_legacy_relation(evidence_api, chain, times=("09:00", "10:00"))
    view = get(evidence_api, chain)
    assert not view["summary"]["latest_record_ids"]
    assert view["coverage"]["relation_proof_unavailable"] == 1
    assert view["coverage"]["unresolved_time_pairs"] > 0
