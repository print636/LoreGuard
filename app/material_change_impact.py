"""Read-only source/version impact for one authorized report, never a verdict."""
from __future__ import annotations

from sqlalchemy import select

from .db import (AnalysisRunRow, AnalysisRunInputRow, AnalysisRunCharacterTraitInputRow,
                 DocumentRow, DocumentContextRow, IssueRow)
from .narrative_context import (context_snapshot_payload, latest_context_revisions,
                                narrative_context_semantic_sha256)
from .service import (document_content_sha256, run_input_metadata,
                      DEFAULT_DOCUMENT_ROLE, DEFAULT_STORY_SCOPE)


def _page(items, offset, limit):
    return dict(items=items[offset:offset + limit], offset=offset, limit=limit,
                total=len(items), has_more=offset + limit < len(items))


def _source(meta):
    return {key: meta.get(key) for key in (
        "document_id", "document_name", "document_version", "document_role",
        "story_scope", "batch_role", "publication_status",
    )}


def material_impact(db, run, *, selected, batch_roles, recheck, comparison_status,
                    traits_changed, offset, limit, issue_offset, issue_limit):
    inputs = db.scalars(select(AnalysisRunInputRow).where(
        AnalysisRunInputRow.run_id == run.id).order_by(AnalysisRunInputRow.ordinal)).all()
    metadata = {m["document_id"]: m for m in run_input_metadata(db, run.id)} if comparison_status == "complete" else {}
    active = db.scalars(select(DocumentRow).where(
        DocumentRow.project_id == run.project_id, DocumentRow.active.is_(True)
    ).order_by(DocumentRow.created_at, DocumentRow.id)).all()
    by_name = {row.name.casefold(): row for row in active}
    contexts = {r.document_id: r for r in db.scalars(select(DocumentContextRow).where(
        DocumentContextRow.document_id.in_([r.id for r in active]))).all()} if active else {}
    revisions = latest_context_revisions(db, [r.id for r in active])
    selected_ids = {row.id for row in selected} if selected is not None else None
    current = {}
    for row in active:
        legacy = contexts.get(row.id)
        role = legacy.document_role if legacy else DEFAULT_DOCUMENT_ROLE
        scope = legacy.story_scope if legacy else DEFAULT_STORY_SCOPE
        narrative = context_snapshot_payload(revisions.get(row.id), document_role=role, story_scope=scope)
        current[row.id] = dict(document_id=row.id, document_name=row.name, document_version=row.version,
            document_role=role, story_scope=scope, batch_role=batch_roles.get(row.id),
            publication_status=narrative.get("publication_status"),
            content_sha256=document_content_sha256(row.content),
            narrative_context_sha256=narrative_context_semantic_sha256(narrative))

    # Sources are frozen input identities, with separately frozen confirmed-trait
    # evidence when that source is not an ordinary input of this run.
    frozen = []
    for row in inputs:
        meta = metadata.get(row.document_id, dict(document_id=row.document_id,
            document_name=row.document_name, document_version=row.document_version,
            content_sha256=row.content_sha256, narrative_context={}))
        frozen.append(dict(key=row.id, kind="input", meta={**meta,
            "publication_status": meta["narrative_context"].get("publication_status")},
            content=row.content, spans=[]))
    seen = {(r["meta"]["document_id"], r["meta"]["content_sha256"]) for r in frozen}
    old_metadata = {}
    for trait in db.scalars(select(AnalysisRunCharacterTraitInputRow).where(
        AnalysisRunCharacterTraitInputRow.run_id == run.id).order_by(AnalysisRunCharacterTraitInputRow.ordinal)).all():
        for evidence in trait.payload.get("evidence", []) if comparison_status == "complete" else []:
            identity = (evidence.get("document_id"), evidence.get("content_sha256"))
            if identity in seen:
                # Additional spans for the same confirmed source stay attached.
                for source in frozen:
                    if source["kind"] == "confirmed_source" and (source["meta"].get("document_id"), source["meta"].get("content_sha256")) == identity:
                        source["spans"].append(evidence)
                continue
            if not all(isinstance(evidence.get(k), str) for k in ("document_id", "document_name", "content_sha256")):
                continue
            seen.add(identity)
            original = db.scalar(select(AnalysisRunInputRow).join(AnalysisRunRow,
                AnalysisRunRow.id == AnalysisRunInputRow.run_id).where(
                    AnalysisRunInputRow.id == evidence.get("input_id"),
                    AnalysisRunRow.project_id == run.project_id))
            original_meta = None
            if original is not None and (original.document_id, original.content_sha256) == identity and document_content_sha256(original.content) == original.content_sha256:
                if original.run_id not in old_metadata:
                    old_metadata[original.run_id] = {m["document_id"]: m for m in run_input_metadata(db, original.run_id)}
                original_meta = old_metadata[original.run_id].get(original.document_id)
            frozen.append(dict(key=f"confirmed:{len(frozen)}", kind="confirmed_source",
                meta={**original_meta, "publication_status": original_meta["narrative_context"].get("publication_status")} if original_meta else evidence,
                content=original.content if original_meta else None, spans=[evidence]))
    source_ids = [s["meta"]["document_id"] for s in frozen]
    existing = set(db.scalars(select(DocumentRow.id).where(
        DocumentRow.project_id == run.project_id, DocumentRow.id.in_(source_ids))).all()) if source_ids else set()
    items, used, affected = [], set(), {}
    for source in frozen:
        old = source["meta"]
        row = by_name.get(old["document_name"].casefold())
        now = current.get(row.id) if row else None
        changes = []
        content_changed = None
        if comparison_status == "unavailable":
            changes.append("unavailable")
        if now is None:
            changes.append("inactive" if old["document_id"] in existing else "missing")
        else:
            used.add(row.id)
            content_changed = now["content_sha256"] != old.get("content_sha256")
            if content_changed: changes.append("content_changed")
            if now["document_version"] != old.get("document_version"): changes.append("version_changed")
            if now["document_id"] != old["document_id"]: changes.append("document_replaced")
            if any(old.get(k) is not None and old.get(k) != now.get(k) for k in (
                "document_role", "story_scope", "narrative_context_sha256")):
                changes.append("identity_changed")
            if source["kind"] == "input" and selected_ids is not None and (
                row.id not in selected_ids or old.get("batch_role") != now.get("batch_role")
            ):
                changes.append("selection_changed")
        item = dict(key=source["key"], kind=source["kind"], frozen=_source(old),
            current=_source(now) if now else None, changes=changes, content_changed=content_changed,
            identity_comparison_available="narrative_context_sha256" in old,
            selected_for_recheck=(row.id in selected_ids if row else False) if selected_ids is not None else None)
        items.append(item)
        if changes and comparison_status == "complete":
            affected[source["key"]] = source
    for row in active:
        if row.id in used: continue
        included = row.id in selected_ids if selected_ids is not None else None
        items.append(dict(key=row.id, kind="added" if included else "not_included", frozen=None,
            current=_source(current[row.id]), changes=[], content_changed=None, selected_for_recheck=included,
            identity_comparison_available=True))

    # No text or private proof is sent back. Evidence must resolve to its frozen
    # source ID and exact canonical lines; a matching name is never a citation.
    issues, unresolved = [], 0
    for issue in db.scalars(select(IssueRow).where(IssueRow.run_id == run.id,
        IssueRow.report_class.in_(["formal", "review_clue"])).order_by(IssueRow.id)).yield_per(100):
        source_keys, indices = [], []
        for index, evidence in enumerate(issue.evidence):
            matches = []
            if comparison_status == "complete" and isinstance(evidence, dict):
                for source in frozen:
                    if evidence.get("document_id") != source["meta"]["document_id"]: continue
                    start, end = evidence.get("line_start"), evidence.get("line_end")
                    if type(start) is not int or type(end) is not int or not 1 <= start <= end: continue
                    if source["content"] is not None:
                        lines = source["content"].splitlines()
                        valid = end <= len(lines) and isinstance(evidence.get("text"), str) and "\n".join(lines[start-1:end]).strip() == evidence["text"].strip()
                    else:
                        valid = any(all(evidence.get(k) == span.get(k) for k in ("document_id", "line_start", "line_end", "text")) for span in source["spans"])
                    if valid: matches.append(source["key"])
            if len(matches) != 1:
                unresolved += 1
                continue
            if matches[0] in affected:
                source_keys.append(matches[0]); indices.append(index)
        if indices:
            issues.append(dict(id=issue.id, title=issue.title, report_class=issue.report_class,
                source_keys=list(dict.fromkeys(source_keys)), evidence_indices=indices))
    return dict(schema_version="material-change-impact-v1", run_id=run.id, project_id=run.project_id,
        comparison_status=comparison_status, semantic_impact_complete=False, confirmed_traits_auto_invalidated=False,
        confirmed_traits_changed=traits_changed, recheck=recheck,
        summary=dict(changed_sources=sum(bool(i["changes"]) for i in items), added_sources=sum(i["kind"] == "added" for i in items),
            not_included_sources=sum(i["kind"] == "not_included" for i in items), affected_issues=len(issues), unresolved_evidence=unresolved,
            unresolved_sources=sum(not i["identity_comparison_available"] for i in items)),
        sources=_page(items, offset, limit), affected_issues=_page(issues, issue_offset, issue_limit))
