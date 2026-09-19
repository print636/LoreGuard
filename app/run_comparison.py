from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any

from sqlalchemy import delete, select

from .db import (
    AnalysisDiagnosticRow,
    AnalysisRunComparisonRow,
    AnalysisRunInputContextRow,
    AnalysisRunInputRow,
    AnalysisRunRow,
    FeedbackRow,
    IssueComparisonItemRow,
    IssueRow,
)
from .time_utils import utc_now_naive


MATCHER_VERSION = "issue-match-v1"
MAX_ISSUES_PER_SIDE = 1_000

# These fields describe the deterministic rule instance rather than generated
# prose or a database id. Values are emitted by LoreGuard's rule engine.
_RULE_IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {
    "fact_conflict": ("subject", "predicate"),
    "location_collision": ("participant", "timestamp"),
    "knowledge_without_acquisition": ("character", "fact"),
    "item_ownership": ("item", "actual_user"),
    "world_rule_conflict": ("key",),
}


def _normalized(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", "", value).casefold()


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rule_identity(issue: IssueRow) -> tuple[str, tuple[str, ...], str] | None:
    fields = _RULE_IDENTITY_FIELDS.get(issue.category)
    if not fields or not isinstance(issue.extra, dict):
        return None
    values = tuple(_normalized(issue.extra.get(field)) for field in fields)
    if not all(values):
        return None
    raw = (issue.category, *values)
    return ("|".join(raw), fields, _digest(raw))


def _evidence_tokens(issue: IssueRow) -> frozenset[str]:
    tokens: set[str] = set()
    rows = issue.evidence if isinstance(issue.evidence, list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = _normalized(row.get("document_name"))
        text = _normalized(row.get("text"))
        if text:
            tokens.add(f"{name}:{text}")
    return frozenset(tokens)


def _evidence_signature(issue: IssueRow) -> str | None:
    tokens = _evidence_tokens(issue)
    return _digest((issue.category, sorted(tokens))) if tokens else None


def _snapshot_rows(db, run_id: str) -> list[AnalysisRunInputRow]:
    return list(
        db.scalars(
            select(AnalysisRunInputRow)
            .where(AnalysisRunInputRow.run_id == run_id)
            .order_by(AnalysisRunInputRow.ordinal, AnalysisRunInputRow.id)
        ).all()
    )


def _snapshot_valid(rows: list[AnalysisRunInputRow]) -> bool:
    if not rows:
        return False
    ordinals: set[int] = set()
    document_ids: set[str] = set()
    logical_names: set[str] = set()
    for row in rows:
        logical_name = row.document_name.casefold()
        if (
            row.ordinal in ordinals
            or row.document_id in document_ids
            or not logical_name
            or logical_name in logical_names
        ):
            return False
        ordinals.add(row.ordinal)
        document_ids.add(row.document_id)
        logical_names.add(logical_name)
        if hashlib.sha256(row.content.encode("utf-8")).hexdigest() != row.content_sha256:
            return False
    return ordinals == set(range(len(rows)))


def build_input_diff(
    baseline_rows: list[AnalysisRunInputRow],
    target_rows: list[AnalysisRunInputRow],
) -> dict[str, Any]:
    baseline = {row.document_name.casefold(): row for row in baseline_rows}
    target = {row.document_name.casefold(): row for row in target_rows}
    old_names = set(baseline)
    new_names = set(target)
    added = sorted(new_names - old_names)
    removed = sorted(old_names - new_names)
    changed = sorted(
        name
        for name in old_names & new_names
        if (
            baseline[name].content_sha256 != target[name].content_sha256
            or baseline[name].document_version != target[name].document_version
        )
    )
    unchanged = sorted((old_names & new_names) - set(changed))
    return {
        "baseline_document_count": len(baseline_rows),
        "target_document_count": len(target_rows),
        "added_documents": added,
        "removed_documents": removed,
        "changed_documents": changed,
        "unchanged_documents": unchanged,
    }


def _input_contexts(db, rows: list[AnalysisRunInputRow]) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for row in rows:
        context = db.get(AnalysisRunInputContextRow, row.id)
        result[row.document_name.casefold()] = (
            context.document_role if context and context.document_role else "chapter",
            context.story_scope if context and context.story_scope else "global",
        )
    return result


def _diagnostic_compatibility(
    baseline: AnalysisDiagnosticRow | None,
    target: AnalysisDiagnosticRow | None,
) -> list[str]:
    reasons: list[str] = []
    payloads: list[tuple[str, dict[str, Any] | None]] = [
        (
            "baseline",
            baseline.payload
            if baseline and isinstance(baseline.payload, dict)
            else None,
        ),
        (
            "target",
            target.payload if target and isinstance(target.payload, dict) else None,
        ),
    ]
    for side, payload in payloads:
        if payload is None:
            reasons.append(f"{side}_diagnostics_missing")
            continue
        model = payload.get("model")
        if not isinstance(model, dict) or not isinstance(
            model.get("partial_fallback"), bool
        ):
            reasons.append(f"{side}_model_diagnostics_incomplete")
        elif model["partial_fallback"]:
            reasons.append(f"{side}_model_partial_fallback")
        investigator = payload.get("evidence_investigator")
        runtime = payload.get("runtime_provenance")
        capabilities = runtime.get("capabilities") if isinstance(runtime, dict) else None
        investigator_enabled = (
            isinstance(capabilities, dict)
            and capabilities.get("evidence_investigator") is True
        )
        if investigator_enabled and (
            not isinstance(investigator, dict)
            or investigator.get("outcome") != "completed"
        ):
            reasons.append(f"{side}_investigator_incomplete")

    if all(payload is not None for _, payload in payloads):
        baseline_runtime = payloads[0][1].get("runtime_provenance")
        target_runtime = payloads[1][1].get("runtime_provenance")
        if not isinstance(baseline_runtime, dict) or not isinstance(
            target_runtime, dict
        ):
            reasons.append("runtime_provenance_missing")
        else:
            # Build hashes remain in diagnostics for audit but are not compared:
            # adding this feature necessarily changes the service bundle. The
            # detection capabilities and provider identity still fail closed.
            for key in ("capabilities", "chat_provider", "rag"):
                if baseline_runtime.get(key) != target_runtime.get(key):
                    reasons.append(f"runtime_{key}_changed")
    return reasons


def _comparison_compatibility(
    db, comparison: AnalysisRunComparisonRow
) -> dict[str, Any]:
    baseline_run = db.get(AnalysisRunRow, comparison.baseline_run_id)
    target_run = db.get(AnalysisRunRow, comparison.target_run_id)
    reasons: list[str] = []
    if baseline_run is None or baseline_run.status != "completed":
        reasons.append("baseline_not_completed")
    if target_run is None or target_run.status != "completed":
        reasons.append("target_not_completed")
    baseline_rows = _snapshot_rows(db, comparison.baseline_run_id)
    target_rows = _snapshot_rows(db, comparison.target_run_id)
    if not _snapshot_valid(baseline_rows):
        reasons.append("baseline_snapshot_incomplete")
    if not _snapshot_valid(target_rows):
        reasons.append("target_snapshot_incomplete")
    input_diff = build_input_diff(baseline_rows, target_rows)
    baseline_contexts = _input_contexts(db, baseline_rows)
    target_contexts = _input_contexts(db, target_rows)
    context_changed = sorted(
        name
        for name in set(baseline_contexts) & set(target_contexts)
        if baseline_contexts[name] != target_contexts[name]
    )
    input_diff["context_changed_documents"] = context_changed
    if input_diff["removed_documents"]:
        reasons.append("documents_removed")
    if context_changed:
        reasons.append("document_context_changed")
    reasons.extend(
        _diagnostic_compatibility(
            db.get(AnalysisDiagnosticRow, comparison.baseline_run_id),
            db.get(AnalysisDiagnosticRow, comparison.target_run_id),
        )
    )
    reasons = sorted(set(reasons))
    return {
        "status": "comparable" if not reasons else "unverifiable",
        "reasons": reasons,
        "input_diff": input_diff,
    }


def _add_item(
    db,
    comparison_id: str,
    outcome: str,
    *,
    baseline: IssueRow | None = None,
    target: IssueRow | None = None,
    method: str | None = None,
    score: int | None = None,
    provenance: dict[str, Any] | None = None,
) -> None:
    db.add(
        IssueComparisonItemRow(
            comparison_id=comparison_id,
            outcome=outcome,
            baseline_issue_id=baseline.id if baseline else None,
            target_issue_id=target.id if target else None,
            match_method=method,
            match_score=score,
            provenance={"matcher_version": MATCHER_VERSION, **(provenance or {})},
        )
    )


def _latest_feedback_snapshots(
    db, issue_ids: set[str]
) -> dict[str, dict[str, Any]]:
    if not issue_ids:
        return {}
    latest: dict[str, FeedbackRow] = {}
    rows = list(
        db.scalars(
            select(FeedbackRow)
            .where(FeedbackRow.issue_id.in_(issue_ids))
            .order_by(FeedbackRow.created_at.desc(), FeedbackRow.id.desc())
        ).all()
    )
    for row in rows:
        latest.setdefault(row.issue_id, row)
    return {
        issue_id: {
            "id": row.id,
            "label": row.label,
            "comment": row.comment,
            "created_at": row.created_at.isoformat(),
        }
        for issue_id, row in latest.items()
    }


def _freeze_feedback_on_items(
    items: list[IssueComparisonItemRow],
    snapshots: dict[str, dict[str, Any]],
) -> None:
    for item in items:
        if item.baseline_issue_id is None:
            continue
        provenance = dict(item.provenance or {})
        provenance["baseline_feedback_snapshot"] = snapshots.get(
            item.baseline_issue_id
        )
        item.provenance = provenance


def _pair_unique_buckets(
    baseline: list[IssueRow], target: list[IssueRow], key
) -> tuple[
    list[tuple[IssueRow, IssueRow, Any]], set[str], set[str], set[str], set[str]
]:
    left: dict[Any, list[IssueRow]] = defaultdict(list)
    right: dict[Any, list[IssueRow]] = defaultdict(list)
    for issue in baseline:
        identity = key(issue)
        if identity is not None:
            left[identity].append(issue)
    for issue in target:
        identity = key(issue)
        if identity is not None:
            right[identity].append(issue)
    pairs: list[tuple[IssueRow, IssueRow, Any]] = []
    used_left: set[str] = set()
    used_right: set[str] = set()
    ambiguous_left: set[str] = set()
    ambiguous_right: set[str] = set()
    for identity in sorted(set(left) & set(right), key=str):
        left_rows = left[identity]
        right_rows = right[identity]
        if len(left_rows) == 1 and len(right_rows) == 1:
            pairs.append((left_rows[0], right_rows[0], identity))
            used_left.add(left_rows[0].id)
            used_right.add(right_rows[0].id)
        else:
            ambiguous_left.update(row.id for row in left_rows)
            ambiguous_right.update(row.id for row in right_rows)
    return pairs, used_left, used_right, ambiguous_left, ambiguous_right


def materialize_run_comparison(
    db, target_run_id: str
) -> AnalysisRunComparisonRow | None:
    """Freeze a conservative one-to-one issue comparison for a revision run."""
    comparison = db.scalar(
        select(AnalysisRunComparisonRow).where(
            AnalysisRunComparisonRow.target_run_id == target_run_id
        )
    )
    if comparison is None:
        return None
    compatibility = _comparison_compatibility(db, comparison)
    baseline_issues = list(
        db.scalars(
            select(IssueRow)
            .where(IssueRow.run_id == comparison.baseline_run_id)
            .order_by(IssueRow.id)
        ).all()
    )
    target_issues = list(
        db.scalars(
            select(IssueRow)
            .where(IssueRow.run_id == target_run_id)
            .order_by(IssueRow.id)
        ).all()
    )
    db.execute(
        delete(IssueComparisonItemRow).where(
            IssueComparisonItemRow.comparison_id == comparison.id
        )
    )

    matched_baseline: set[str] = set()
    matched_target: set[str] = set()
    ambiguous_baseline: set[str] = set()
    ambiguous_target: set[str] = set()

    if (
        len(baseline_issues) > MAX_ISSUES_PER_SIDE
        or len(target_issues) > MAX_ISSUES_PER_SIDE
    ):
        compatibility["status"] = "unverifiable"
        compatibility["reasons"] = sorted(
            set(compatibility["reasons"] + ["issue_limit_exceeded"])
        )
        ambiguous_baseline.update(row.id for row in baseline_issues)
        ambiguous_target.update(row.id for row in target_issues)
    else:
        rule_pairs, left_used, right_used, left_ambiguous, right_ambiguous = (
            _pair_unique_buckets(
                baseline_issues,
                target_issues,
                key=lambda issue: (
                    (issue.category, identity[0])
                    if (identity := _rule_identity(issue)) is not None
                    else None
                ),
            )
        )
        matched_baseline.update(left_used)
        matched_target.update(right_used)
        ambiguous_baseline.update(left_ambiguous)
        ambiguous_target.update(right_ambiguous)
        for baseline, target, _ in rule_pairs:
            rule = _rule_identity(baseline)
            left_evidence = _evidence_tokens(baseline)
            right_evidence = _evidence_tokens(target)
            union = len(left_evidence | right_evidence)
            overlap = len(left_evidence & right_evidence) / union if union else 0.0
            _add_item(
                db,
                comparison.id,
                "persisting",
                baseline=baseline,
                target=target,
                method="rule_identity",
                score=100 + round(overlap * 30),
                provenance={
                    "category": baseline.category,
                    "rule_fields": list(rule[1]),
                    "rule_identity_sha256": rule[2],
                    "evidence_overlap": round(overlap, 4),
                },
            )

        evidence_baseline = [
            row
            for row in baseline_issues
            if row.id not in matched_baseline
            and row.id not in ambiguous_baseline
        ]
        evidence_target = [
            row
            for row in target_issues
            if row.id not in matched_target
            and row.id not in ambiguous_target
        ]
        evidence_pairs, _, _, left_ambiguous, right_ambiguous = (
            _pair_unique_buckets(
                evidence_baseline,
                evidence_target,
                key=lambda issue: (
                    (issue.category, signature)
                    if (signature := _evidence_signature(issue)) is not None
                    else None
                ),
            )
        )
        ambiguous_baseline.update(left_ambiguous)
        ambiguous_target.update(right_ambiguous)
        for baseline, target, identity in evidence_pairs:
            baseline_rule = _rule_identity(baseline)
            target_rule = _rule_identity(target)
            if (
                baseline_rule is not None
                and target_rule is not None
                and baseline_rule[0] != target_rule[0]
            ):
                ambiguous_baseline.add(baseline.id)
                ambiguous_target.add(target.id)
                continue
            matched_baseline.add(baseline.id)
            matched_target.add(target.id)
            _add_item(
                db,
                comparison.id,
                "persisting",
                baseline=baseline,
                target=target,
                method="evidence_signature",
                score=75,
                provenance={
                    "category": baseline.category,
                    "evidence_sha256": identity[1],
                    "evidence_overlap": 1.0,
                },
            )

    for issue in baseline_issues:
        if issue.id in matched_baseline:
            continue
        if issue.id in ambiguous_baseline:
            outcome = "unverifiable"
            reason = "ambiguous_identity"
        elif compatibility["status"] != "comparable":
            outcome = "unverifiable"
            reason = "runs_not_comparable"
        else:
            outcome = "no_longer_detected"
            reason = "no_matching_issue_detected"
        _add_item(
            db,
            comparison.id,
            outcome,
            baseline=issue,
            provenance={"reason": reason},
        )
    for issue in target_issues:
        if issue.id in matched_target:
            continue
        if issue.id in ambiguous_target:
            outcome = "unverifiable"
            reason = "ambiguous_identity"
        elif compatibility["status"] != "comparable":
            outcome = "unverifiable"
            reason = "runs_not_comparable"
        else:
            outcome = "new"
            reason = "no_matching_baseline_issue"
        _add_item(
            db,
            comparison.id,
            outcome,
            target=issue,
            provenance={"reason": reason},
        )

    counts = {
        name: 0
        for name in ("no_longer_detected", "persisting", "new", "unverifiable")
    }
    # Added ORM rows are visible in session.new before the flush.
    for row in db.new:
        if (
            isinstance(row, IssueComparisonItemRow)
            and row.comparison_id == comparison.id
        ):
            counts[row.outcome] += 1
    db.flush()
    item_rows = list(
        db.scalars(
            select(IssueComparisonItemRow).where(
                IssueComparisonItemRow.comparison_id == comparison.id
            )
        ).all()
    )
    baseline_issue_ids = {row.id for row in baseline_issues}
    feedback_snapshots = _latest_feedback_snapshots(db, baseline_issue_ids)
    _freeze_feedback_on_items(item_rows, feedback_snapshots)
    no_longer_ids = {
        row.baseline_issue_id
        for row in item_rows
        if row.outcome == "no_longer_detected" and row.baseline_issue_id is not None
    }
    baseline_false_positive = sum(
        1
        for issue_id in baseline_issue_ids
        if feedback_snapshots.get(issue_id, {}).get("label") == "false_positive"
    )
    no_longer_false_positive = sum(
        1
        for issue_id in no_longer_ids
        if feedback_snapshots.get(issue_id, {}).get("label") == "false_positive"
    )
    comparison.matcher_version = MATCHER_VERSION
    comparison.status = "ready"
    comparison.summary = {
        **counts,
        "actionable_no_longer_detected": (
            counts["no_longer_detected"] - no_longer_false_positive
        ),
        "baseline_false_positive": baseline_false_positive,
        "total_baseline": len(baseline_issues),
        "total_target": len(target_issues),
    }
    comparison.provenance = {
        "matcher_version": MATCHER_VERSION,
        "compatibility": compatibility,
        "semantic_equivalence_guaranteed": False,
        "issue_limit_per_side": MAX_ISSUES_PER_SIDE,
    }
    comparison.completed_at = utc_now_naive()
    db.flush()
    return comparison


def mark_comparison_unverifiable(
    db, target_run_id: str, reason: str = "comparison_internal_error"
) -> AnalysisRunComparisonRow | None:
    """Produce an explicit fail-closed report after an unexpected matcher error."""
    comparison = db.scalar(
        select(AnalysisRunComparisonRow).where(
            AnalysisRunComparisonRow.target_run_id == target_run_id
        )
    )
    if comparison is None:
        return None
    baseline_issues = list(
        db.scalars(
            select(IssueRow).where(IssueRow.run_id == comparison.baseline_run_id)
        ).all()
    )
    target_issues = list(
        db.scalars(select(IssueRow).where(IssueRow.run_id == target_run_id)).all()
    )
    db.execute(
        delete(IssueComparisonItemRow).where(
            IssueComparisonItemRow.comparison_id == comparison.id
        )
    )
    for issue in baseline_issues:
        _add_item(
            db,
            comparison.id,
            "unverifiable",
            baseline=issue,
            provenance={"reason": reason},
        )
    for issue in target_issues:
        _add_item(
            db,
            comparison.id,
            "unverifiable",
            target=issue,
            provenance={"reason": reason},
        )
    db.flush()
    item_rows = list(
        db.scalars(
            select(IssueComparisonItemRow).where(
                IssueComparisonItemRow.comparison_id == comparison.id
            )
        ).all()
    )
    baseline_issue_ids = {row.id for row in baseline_issues}
    feedback_snapshots = _latest_feedback_snapshots(db, baseline_issue_ids)
    _freeze_feedback_on_items(item_rows, feedback_snapshots)
    baseline_false_positive = sum(
        1
        for issue_id in baseline_issue_ids
        if feedback_snapshots.get(issue_id, {}).get("label") == "false_positive"
    )
    comparison.status = "ready"
    comparison.summary = {
        "no_longer_detected": 0,
        "persisting": 0,
        "new": 0,
        "unverifiable": len(baseline_issues) + len(target_issues),
        "actionable_no_longer_detected": 0,
        "baseline_false_positive": baseline_false_positive,
        "total_baseline": len(baseline_issues),
        "total_target": len(target_issues),
    }
    comparison.provenance = {
        "matcher_version": MATCHER_VERSION,
        "compatibility": {"status": "unverifiable", "reasons": [reason]},
        "semantic_equivalence_guaranteed": False,
        "issue_limit_per_side": MAX_ISSUES_PER_SIDE,
    }
    comparison.completed_at = utc_now_naive()
    db.flush()
    return comparison
