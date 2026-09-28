"""Require complete proof for persisted formal character issues.

Revision ID: 0022_character_formal_proof
Revises: 0021_issue_report_classes

Older character rows predate the two-current-event, event-identity and
explanation-coverage contracts.  A row without every persisted proof is kept
for audit but conservatively moved to the separate review-clue surface.
"""

from __future__ import annotations

import re
import unicodedata

import sqlalchemy as sa
from alembic import op


revision = "0022_character_formal_proof"
down_revision = "0021_issue_report_classes"
branch_labels = None
depends_on = None


_CITATION_HANDLE = re.compile(r"^(?P<role>[BCGXP])(?P<number>0[1-9]|[1-9][0-9])$")
_HANDLE_LIMITS = {"B": 12, "C": 6, "G": 16, "X": 16, "P": 16}
_ROLE_ORDER = "BCGXP"


def _validated_span(span: object) -> tuple[str, int, int, str] | None:
    if not isinstance(span, dict):
        return None
    document_id = span.get("document_id")
    line_start = span.get("line_start")
    line_end = span.get("line_end")
    text = span.get("text")
    if (
        not isinstance(document_id, str)
        or not document_id.strip()
        or type(line_start) is not int
        or type(line_end) is not int
        or not 1 <= line_start <= line_end
        or not isinstance(text, str)
    ):
        return None
    normalized_text = " ".join(unicodedata.normalize("NFKC", text).split())
    if not normalized_text:
        return None
    return document_id, line_start, line_end, normalized_text


def _complete_formal_proof(extra: object, evidence: object) -> bool:
    if not isinstance(extra, dict) or not isinstance(evidence, list):
        return False
    if (
        extra.get("final_outcome") != "conflict"
        or extra.get("evidence_binding") != "review_citations_v1"
        or extra.get("event_independence") != "yes"
        or extra.get("event_identity_verification") != "different_events"
        or extra.get("material_coverage") != "complete"
        or extra.get("explanation_coverage") != "complete"
        or extra.get("explanation_review_executed") is not True
    ):
        return False
    selected = extra.get("independent_event_citations")
    refs = extra.get("review_citation_refs")
    if (
        not isinstance(selected, list)
        or len(selected) != 2
        or not all(isinstance(value, str) for value in selected)
        or len(set(selected)) != 2
        or not isinstance(refs, list)
        or not 1 <= len(refs) <= 8
    ):
        return False

    baseline_count = 0
    current: dict[str, tuple[str, int, int, str]] = {}
    seen_handles: set[str] = set()
    referenced_evidence_indexes: set[int] = set()
    roles_by_evidence_index: dict[int, set[str]] = {}
    validated_refs: list[tuple[str, str, int]] = []
    for ref_position, ref in enumerate(refs):
        if not isinstance(ref, dict):
            return False
        handle = ref.get("handle")
        role = ref.get("role")
        evidence_index = ref.get("evidence_index")
        response_index = ref.get("response_index")
        handle_match = (
            _CITATION_HANDLE.fullmatch(handle)
            if isinstance(handle, str)
            else None
        )
        if (
            handle_match is None
            or role not in {"B", "C", "G", "X", "P"}
            or handle_match.group("role") != role
            or int(handle_match.group("number")) > _HANDLE_LIMITS[role]
            or handle in seen_handles
            or type(response_index) is not int
            or response_index != ref_position
            or type(evidence_index) is not int
            or not 0 <= evidence_index < len(evidence)
        ):
            return False
        span = _validated_span(evidence[evidence_index])
        if span is None:
            return False
        seen_handles.add(handle)
        referenced_evidence_indexes.add(evidence_index)
        roles_by_evidence_index.setdefault(evidence_index, set()).add(role)
        validated_refs.append((handle, role, evidence_index))
        if role == "B":
            baseline_count += 1
        elif role == "C":
            current[handle] = span
    if (
        baseline_count < 1
        or set(current) != set(selected)
        or referenced_evidence_indexes != set(range(len(evidence)))
        or any(
            "B" in roles and len(roles) != 1
            for roles in roles_by_evidence_index.values()
        )
    ):
        return False

    # Production persists evidence in canonical role/handle order while refs
    # remain in model-response order. Rebuild that ordering so a forged handle
    # cannot simply point at another role's otherwise valid span.
    next_evidence_index = 0
    seen_ordered_indexes: set[int] = set()
    for handle, _, evidence_index in sorted(
        validated_refs,
        key=lambda row: (_ROLE_ORDER.index(row[0][0]), int(row[0][1:])),
    ):
        if evidence_index in seen_ordered_indexes:
            continue
        if evidence_index != next_evidence_index:
            return False
        seen_ordered_indexes.add(evidence_index)
        next_evidence_index += 1

    left, right = (current[handle] for handle in selected)
    if left[3] == right[3]:
        return False
    if (
        left[0] == right[0]
        and max(left[1], right[1]) <= min(left[2], right[2])
    ):
        return False
    return True


def upgrade() -> None:
    bind = op.get_bind()
    table_names = set(sa.inspect(bind).get_table_names())
    if "issues" not in table_names or "report_class" not in {
        row["name"] for row in sa.inspect(bind).get_columns("issues")
    }:
        return

    issues = sa.table(
        "issues",
        sa.column("id", sa.String(36)),
        sa.column("run_id", sa.String(36)),
        sa.column("category", sa.String(80)),
        sa.column("evidence", sa.JSON()),
        sa.column("extra", sa.JSON()),
        sa.column("report_class", sa.String(20)),
    )
    rows = bind.execute(
        sa.select(
            issues.c.id,
            issues.c.run_id,
            issues.c.evidence,
            issues.c.extra,
        ).where(
            issues.c.category == "character_drift",
            issues.c.report_class == "formal",
        )
    ).mappings().all()
    demoted_run_ids: set[str] = set()
    for row in rows:
        if _complete_formal_proof(row["extra"], row["evidence"]):
            continue
        prior_extra = row["extra"] if isinstance(row["extra"], dict) else {}
        previous_outcome = prior_extra.get("final_outcome")
        extra = {
            **prior_extra,
            "final_outcome": "needs_confirmation",
            "review_reason": "legacy_formal_proof_incomplete",
            "legacy_formal_proof_incomplete": True,
            **(
                {"legacy_previous_final_outcome": previous_outcome}
                if isinstance(previous_outcome, str) else {}
            ),
        }
        bind.execute(
            sa.update(issues)
            .where(issues.c.id == row["id"])
            .values(report_class="review_clue", extra=extra)
        )
        demoted_run_ids.add(row["run_id"])

    if (
        demoted_run_ids
        and {"analysis_run_comparisons", "issue_comparison_items"}
        <= table_names
    ):
        comparisons = sa.table(
            "analysis_run_comparisons",
            sa.column("id", sa.String(36)),
            sa.column("baseline_run_id", sa.String(36)),
            sa.column("target_run_id", sa.String(36)),
            sa.column("status", sa.String(24)),
            sa.column("summary", sa.JSON()),
            sa.column("provenance", sa.JSON()),
            sa.column("completed_at", sa.DateTime()),
        )
        comparison_items = sa.table(
            "issue_comparison_items",
            sa.column("comparison_id", sa.String(36)),
        )
        affected = sa.select(comparisons.c.id).where(
            sa.or_(
                comparisons.c.baseline_run_id.in_(demoted_run_ids),
                comparisons.c.target_run_id.in_(demoted_run_ids),
            )
        )
        bind.execute(
            sa.delete(comparison_items).where(
                comparison_items.c.comparison_id.in_(affected)
            )
        )
        bind.execute(
            sa.update(comparisons)
            .where(comparisons.c.id.in_(affected))
            .values(status="pending", summary={}, provenance={}, completed_at=None)
        )


def downgrade() -> None:
    # The previous schema cannot prove which historical rows were formal under
    # the stronger contract. Re-promoting them would recreate false reports,
    # so this data-only safety migration is intentionally irreversible.
    return
