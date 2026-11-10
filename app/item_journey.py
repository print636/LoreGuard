"""Read-only, run-scoped physical item histories from persisted evidence.

Names are search groups, not new entity identities. A terminal record is an
attestation at that event, never a claim about continuous or present ownership.
"""
from __future__ import annotations

from sqlalchemy import func, select

from .db import (AnalysisDiagnosticRow, AnalysisRecordRow, AnalysisRunInputContextRow,
                 AnalysisRunInputRow, IssueRow)
from .domain import ParsedDirective
from .narrative_relations import _absolute, _clock, compare_time_points
from .narrative_time_scope import TIME_SCOPE_REFS_KEY, TIME_SCOPE_SCHEMA, member_matches
from .narrative_time_scope_storage import time_scope_storage_payload
from .projections import record_sort_key
from .report_evidence_preview import EvidencePreviewUnavailable, _reference, _verified_lines
from .rules import _scopes_compatible
from .semantic_quality import document_context_has_noncanonical_frame, eligible_for_deterministic_rules
from .semantic_state import SEMANTIC_STATE_REFS_KEY, SEMANTIC_STATE_SCHEMA, digest
from .semantic_state_storage import semantic_state_storage_payload


SCHEMA = "item-journey-v1"
MAX_CATALOG_RECORDS = 10_000
MAX_JOURNEY_RECORDS = 200
MAX_SCOPE_RECORDS = 128


class ItemJourneyNotFound(Exception):
    pass


def _key(run_id, name, scope):
    return "ij_" + digest([run_id, name, scope])[:32]


def _group(row):
    attrs = row.attrs if isinstance(row.attrs, dict) else {}
    name, scope = attrs.get("item"), attrs.get("story_scope", "global") or "global"
    if not isinstance(name, str) or not name.strip() or not isinstance(scope, str):
        return None
    return name, scope


def _possibly_same_time(left, right):
    """Only parsed incompatibility can exclude an uncertified endpoint.

    Prefixes and narrative spellings may be equivalent. Failure to parse is
    not a different time, and never becomes a positive order certificate.
    """
    if not isinstance(left, str) or not isinstance(right, str):
        return True
    for parser in (_absolute, _clock):
        a, b = parser(left), parser(right)
        if a is not None and b is not None:
            return a == b
    return True


def _catalog(db, run):
    query = select(AnalysisRecordRow).where(AnalysisRecordRow.run_id == run.id,
                                          AnalysisRecordRow.kind.in_(("item", "uses")))
    total = db.scalar(select(func.count()).select_from(query.subquery()))
    rows = list(db.scalars(query.order_by(AnalysisRecordRow.id).limit(MAX_CATALOG_RECORDS)).all())
    groups = {}
    for row in rows:
        group = _group(row)
        if group:
            groups.setdefault(group, []).append(row)
    return groups, total, len(rows)


def _issue_matches(issue, name, rows):
    """No title/name-only jump: require the issue's actual frozen evidence."""
    if not isinstance(issue.extra, dict) or issue.extra.get("item") != name:
        return False
    spans = issue.evidence if isinstance(issue.evidence, list) else []
    return any(row.evidence == span for row in rows for span in spans)


def _item(run, group, rows, issues):
    name, scope = group
    return dict(key=_key(run.id, name, scope), name=name, story_scope=scope, record_count=len(rows),
                issue_ids=[issue.id for issue in issues if _issue_matches(issue, name, rows)])


def list_item_journeys(db, run, *, q="", issue_id=None, offset=0, limit=30):
    groups, total, scanned = _catalog(db, run)
    query = select(IssueRow).where(IssueRow.run_id == run.id, IssueRow.category == "item_ownership")
    issues = list(db.scalars(query).all())
    if issue_id is not None and not any(row.id == issue_id for row in issues):
        raise ItemJourneyNotFound
    items = [_item(run, group, rows, issues) for group, rows in groups.items()
             if q.casefold() in group[0].casefold()]
    if issue_id is not None:
        items = [item for item in items if issue_id in item["issue_ids"]]
    items.sort(key=lambda item: (item["name"].casefold(), item["story_scope"], item["key"]))
    return dict(schema_version=SCHEMA, run_id=run.id, project_id=run.project_id,
                items=items[offset:offset + limit],
                page=dict(offset=offset, limit=limit, total=len(items), has_more=offset + limit < len(items)),
                coverage=dict(scanned_records=scanned, total_records=total, truncated=total > scanned))


def _receipts(payload, key, schema):
    envelope = payload.get(key)
    if envelope is None:
        return {}
    if (not isinstance(envelope, dict) or set(envelope) != {"schema_version", "receipts", "payload_sha256"}
            or envelope.get("schema_version") != schema or not isinstance(envelope.get("receipts"), list)
            or envelope.get("payload_sha256") != digest({k: v for k, v in envelope.items() if k != "payload_sha256"})):
        return {}
    rows = envelope["receipts"]
    if any(not isinstance(row, dict) or not isinstance(row.get("record_id"), str) for row in rows):
        return {}
    result = {row["record_id"]: row for row in rows}
    return result if len(result) == len(rows) else {}


class _Snapshot:
    def __init__(self, db, run):
        self.db, self.run = db, run
        diagnostic = db.get(AnalysisDiagnosticRow, run.id)
        self.payload = diagnostic.payload if diagnostic and isinstance(diagnostic.payload, dict) else {}
        self.state_refs = _receipts(self.payload, SEMANTIC_STATE_REFS_KEY, SEMANTIC_STATE_SCHEMA)
        self.scope_refs = _receipts(self.payload, TIME_SCOPE_REFS_KEY, TIME_SCOPE_SCHEMA)
        self.sources = {source.document_id: source for source in db.scalars(select(AnalysisRunInputRow).where(
            AnalysisRunInputRow.run_id == run.id)).all()}
        self.contexts = {source.id: db.get(AnalysisRunInputContextRow, source.id) for source in self.sources.values()}
        self.cache = {}

    def load(self, record):
        """Source readability and semantic acceptance are deliberately separate."""
        if record.id in self.cache:
            return self.cache[record.id]
        row = source = None
        supported = False
        try:
            reference = _reference(record.evidence)
            source = self.sources.get(reference["document_id"])
            if source is None:
                raise ValueError
            _verified_lines(source, reference)
            if record.run_id != self.run.id:
                raise ValueError
            context = self.contexts[source.id]
            if context is not None and any(getattr(context, k) is not None and record.attrs.get(k) != getattr(context, k)
                                           for k in ("document_role", "story_scope")):
                raise ValueError
            row = ParsedDirective(kind=record.kind, attrs=record.attrs, evidence=reference,
                noncanonical_frame=document_context_has_noncanonical_frame(
                    source.content, reference["line_start"], reference["line_end"]))
        except (EvidencePreviewUnavailable, ValueError, TypeError, KeyError):
            row = source = None
        if row is not None and record.id in self.state_refs:
            receipt = self.state_refs[record.id]
            try:
                candidate = row.model_copy(update={"semantic_state": receipt["binding"]})
                rebuilt = semantic_state_storage_payload(self.db, run=self.run, records=[record], directives=[candidate])
                if rebuilt is not None and rebuilt["receipts"] == [receipt]:
                    row = candidate
                    supported = eligible_for_deterministic_rules(row)
            except (ValueError, TypeError, KeyError, AttributeError, IndexError):
                pass  # Read old/damaged proof as unavailable, never re-certify it.
        self.cache[record.id] = (row, source, supported)
        return self.cache[record.id]

    def scopes(self, selected_records):
        all_pairs = {frozenset((left.id, right.id)) for index, left in enumerate(selected_records)
                     for right in selected_records[index + 1:]}
        self.blocked_pairs = set()
        total = self.db.scalar(select(func.count()).select_from(AnalysisRecordRow).where(
            AnalysisRecordRow.run_id == self.run.id,
            AnalysisRecordRow.kind.in_(("narrative_time_scope", "narrative_time_relation"))))
        records = list(self.db.scalars(select(AnalysisRecordRow).where(
            AnalysisRecordRow.run_id == self.run.id,
            AnalysisRecordRow.kind.in_(("narrative_time_scope", "narrative_time_relation")))
            .order_by(AnalysisRecordRow.id).limit(MAX_SCOPE_RECORDS)).all())
        valid, unavailable = [], 0
        for record in records:
            row, source, _ = self.load(record)
            relevant_pairs = all_pairs
            selected = [(actual, self.load(actual)[0]) for actual in selected_records]
            if row is not None:
                # Only a verified frozen scope can exclude a different branch.
                selected = [(actual, parsed) for actual, parsed in selected
                            if parsed is None or _scopes_compatible(row, parsed)]
                relevant_pairs = {frozenset((a.id, b.id)) for index, (a, _) in enumerate(selected)
                                  for b, _ in selected[index + 1:]}
                if record.kind == "narrative_time_relation":
                    left_time, right_time = row.attrs.get("left_time"), row.attrs.get("right_time")
                    if left_time and right_time:
                        # Legacy direct relations retained original endpoint
                        # times, not their source identities. They cannot prove
                        # an order. They can only flag pairs they could affect;
                        # absent/ambiguous endpoints remain conservative.
                        relevant_pairs = {frozenset((a.id, b.id)) for a, pa in selected for b, pb in selected
                                          if a.id != b.id and (pa is None or _possibly_same_time(pa.attrs.get("time"), left_time))
                                          and (pb is None or _possibly_same_time(pb.attrs.get("time"), right_time))}
            if not relevant_pairs:
                continue
            receipt = self.scope_refs.get(record.id)
            if record.kind != "narrative_time_scope" or row is None or not isinstance(receipt, dict):
                unavailable += 1
                self.blocked_pairs.update(relevant_pairs)
                continue
            # Validation failure must retain a barrier for the potentially
            # affected pairs. It must not poison verified unrelated branches.
            accepted = False
            try:
                row = row.model_copy(update={"narrative_time_scope": receipt["binding"]})
                # Reuse the original save-time validator, including source frames
                # and each matched endpoint. Only SELECTs occur here.
                matched_ids = {rid for member in receipt["members"] for rid in member["matched_record_ids"]}
                if len(matched_ids) > MAX_CATALOG_RECORDS:
                    continue
                actual = list(self.db.scalars(select(AnalysisRecordRow).where(
                    AnalysisRecordRow.run_id == self.run.id, AnalysisRecordRow.id.in_(matched_ids))).all())
                if {record.id for record in actual} != matched_ids:
                    continue
                directives = [self.load(actual_row)[0] for actual_row in actual]
                if any(directive is None for directive in directives):
                    continue
                # A selected record cannot borrow a different endpoint's stored
                # match, even if its time or physical line happens to coincide.
                for candidate in selected_records:
                    parsed = self.load(candidate)[0]
                    if parsed is None:
                        continue
                    for index, member in enumerate(row.narrative_time_scope["members"]):
                        if member_matches(member, parsed) != (candidate.id in receipt["members"][index]["matched_record_ids"]):
                            raise ValueError
                rebuilt = time_scope_storage_payload(self.db, run=self.run, records=[*actual, record],
                                                     directives=[*directives, row])
                new = rebuilt["receipts"][0]
                # Ordering of matched IDs is not semantic; original extraction
                # order and SELECT order need not be identical.
                for value in (new,):
                    for member in value["members"]:
                        member["matched_record_ids"] = sorted(member["matched_record_ids"])
                expected = {**receipt, "members": [{**m, "matched_record_ids": sorted(m["matched_record_ids"])}
                                                   for m in receipt["members"]]}
                if new != expected:
                    continue
                valid.append((record, row, source))
                accepted = True
            except (ValueError, TypeError, KeyError, AttributeError, IndexError):
                pass
            finally:
                if not accepted:
                    unavailable += 1
                    self.blocked_pairs.update(relevant_pairs)
        # Unread records at the bound have unknown relevance, not assumed
        # irrelevance. No terminal inference may depend on their omission.
        if total > len(records):
            unavailable += total - len(records)
            self.blocked_pairs.update(all_pairs)
        return valid, unavailable


def _event(record, snapshot):
    row, source, supported = snapshot.load(record)
    attrs = record.attrs if isinstance(record.attrs, dict) else {}
    kind = "uses" if record.kind == "uses" else attrs.get("state_kind", "legacy")
    if kind not in {"observed_possession", "custody", "transfer", "retained_custody", "uses"}:
        kind = "legacy"
    # Uncertified state labels remain visible as record descriptions, with an
    # explicit unverified status; they never participate in the summary.
    return dict(record_id=record.id, kind=kind, holder=attrs.get("user" if record.kind == "uses" else "owner") or None,
                transfer_from=attrs.get("transfer_from") or None, time=attrs.get("time") or None,
                through_time=attrs.get("through_time") or None, retention_scope=attrs.get("retention_scope") or None,
                basis=row.semantic_state["basis"] if supported else "unverified",
                source_status="verified" if source else "unavailable", semantic_status="supported" if supported else "unverified",
                evidence=row.evidence.model_dump(mode="json") if row else None,
                document_version=source.document_version if source else None)


def _description(event):
    time = f"{event['time']}，" if event["time"] else "该事件中，"
    holder = event["holder"] or "未确定的人物"
    if event["kind"] == "transfer":
        action = f"{event['transfer_from']}将物品交给{holder}"
    elif event["kind"] == "uses":
        action = f"{holder}使用该物品"
    elif event["kind"] == "observed_possession":
        action = f"{holder}持有该物品"
    else:
        action = f"{holder}保管该物品"
    return time + action


def get_item_journey(db, run, item_key, *, offset=0, limit=50):
    groups, all_count, scanned = _catalog(db, run)
    group = next((group for group in groups if _key(run.id, *group) == item_key), None)
    if group is None:
        raise ItemJourneyNotFound
    all_rows = groups[group]
    all_rows.sort(key=record_sort_key)
    records = all_rows[:MAX_JOURNEY_RECORDS]
    snapshot = _Snapshot(db, run)
    events = [_event(row, snapshot) for row in records]
    scopes, unavailable_relations = snapshot.scopes(records)
    relations, unknown, orders = [], 0, {}
    for index, left in enumerate(records):
        a, _, supported_a = snapshot.load(left)
        for right in records[index + 1:]:
            b, _, supported_b = snapshot.load(right)
            order, evidence = (compare_time_points(a, b, [scope[1] for scope in scopes])
                               if supported_a and supported_b
                               and frozenset((left.id, right.id)) not in getattr(snapshot, "blocked_pairs", ())
                               else (None, []))
            orders[left.id, right.id] = order
            orders[right.id, left.id] = -order if order is not None else None
            if order is None:
                unknown += 1
            else:
                evidence_ids = [record.id for record, scope, _ in scopes if scope.evidence in evidence]
                relations.append(dict(left_record_id=left.id, right_record_id=right.id,
                                      order={-1: "before", 0: "same_time", 1: "after"}[order],
                                      evidence_record_ids=evidence_ids))
    source_missing = sum(event["source_status"] != "verified" for event in events)
    proof_missing = sum(event["semantic_status"] != "supported" for event in events)
    truncated = len(all_rows) > len(records) or all_count > scanned
    latest = []
    if not (truncated or source_missing or proof_missing):
        latest = [event for event in events if all(orders.get((event["record_id"], other["record_id"])) in {0, 1}
                  for other in events if other is not event)]
    if latest:
        text = "这次审查最后有据记录：" + "；".join(_description(event) for event in latest[:3]) + "。"
        if len(latest) > 3:
            text += f"同一末端时点还有{len(latest) - 3}条记录。"
        text += "这不代表此后持续保管，也不判断法律所有权。"
        if unknown:
            text += f"另有{unknown}组记录的相互先后仍未明，未据此补成连续保管链。"
        status = "ordered_evidence"
    elif any(event["semantic_status"] == "supported" for event in events):
        text = "已有有源持有、交接或使用记录；因记录覆盖或先后关系未明，不能确定最后持有人。"
        status = "partial"
    else:
        text = "仅有尚不能核验完整语义依据的记录，不能据此确定持有人或交接历程。"
        status = "unavailable"
    text += "仅针对这次审查已保留的记录，不代表全文事件已找全。同名记录按报告内名称与范围分组，不自动证明均为同一实物。"
    issues = list(db.scalars(select(IssueRow).where(IssueRow.run_id == run.id,
                                                IssueRow.category == "item_ownership")).all())
    used = {record_id for relation in relations for record_id in relation["evidence_record_ids"]}
    model = snapshot.payload.get("model", {})
    run_partial = (run.status != "completed" or not isinstance(model, dict)
                   or model.get("partial_fallback") is not False)
    return dict(schema_version=SCHEMA, run_id=run.id, project_id=run.project_id,
                item=_item(run, group, all_rows, issues),
                summary=dict(status=status, text=text, latest_record_ids=[event["record_id"] for event in latest]),
                events=dict(items=events[offset:offset + limit], offset=offset, limit=limit,
                            total=len(all_rows), has_more=offset + limit < len(events)),
                relations=relations,
                relation_evidence=[dict(record_id=record.id, evidence=row.evidence.model_dump(mode="json"),
                                        document_version=source.document_version)
                                   for record, row, source in scopes if record.id in used],
                coverage=dict(processed_records=len(records), total_records=len(all_rows), truncated=truncated,
                              source_unavailable=source_missing, proof_unavailable=proof_missing,
                              unresolved_time_pairs=unknown, run_partial=run_partial,
                              relation_proof_unavailable=unavailable_relations))
