"""Keep review clues out of the formal issue report.

Revision ID: 0021_issue_report_classes
Revises: 0020_value_boundary_author_axes

Historical character rows did not store the final promotion outcome.  Only
rows matching the old, fixed conflict presentation and carrying a direct
opposition verdict remain formal.  Ambiguous rows are kept as review clues;
their existing feedback remains in the database for audit.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0021_issue_report_classes"
down_revision = "0020_value_boundary_author_axes"
branch_labels = None
depends_on = None


def _historical_formal_character_issue(row: sa.RowMapping) -> bool:
    extra = row["extra"]
    evidence = row["evidence"]
    if (
        not isinstance(extra, dict)
        or extra.get("judgement") not in {"contradicts", "deterministic_conflict"}
        or row["severity"] != "high"
        or not isinstance(row["title"], str)
        or not row["title"].endswith("的角色设定可能冲突")
        or not isinstance(evidence, list)
        or len(evidence) < 2
    ):
        return False
    coordinates: list[tuple[str, int, int]] = []
    for span in evidence[:2]:
        if not isinstance(span, dict):
            return False
        document_id = span.get("document_id")
        line_start = span.get("line_start")
        line_end = span.get("line_end")
        if (
            not isinstance(document_id, str) or not document_id
            or type(line_start) is not int or type(line_end) is not int
            or not 1 <= line_start <= line_end
        ):
            return False
        coordinates.append((document_id, line_start, line_end))
    return coordinates[0] != coordinates[1]


def upgrade() -> None:
    bind = op.get_bind()
    table_names = set(sa.inspect(bind).get_table_names())
    if "issues" not in table_names:
        # A stamped, partial pre-Alembic installation may only contain the
        # embedding substrate.  Preserve the existing adoption convention:
        # there is no issue report to migrate until that table exists.
        return
    if "report_class" not in {
        row["name"] for row in sa.inspect(bind).get_columns("issues")
    }:
        # Product init_db may have created the current ORM schema before
        # Alembic adopts/stamps the database.  Do not add its column twice.
        op.add_column(
            "issues",
            sa.Column(
                "report_class", sa.String(20), nullable=False,
                server_default="formal",
            ),
        )
    issues = sa.table(
        "issues",
        sa.column("id", sa.String(36)),
        sa.column("run_id", sa.String(36)),
        sa.column("category", sa.String(80)),
        sa.column("severity", sa.String(20)),
        sa.column("title", sa.String(255)),
        sa.column("evidence", sa.JSON()),
        sa.column("extra", sa.JSON()),
        sa.column("report_class", sa.String(20)),
    )
    old_character_rows = bind.execute(
        sa.select(
            issues.c.id, issues.c.category, issues.c.severity,
            issues.c.title, issues.c.evidence, issues.c.extra,
            issues.c.report_class,
        ).where(issues.c.category == "character_drift")
    ).mappings()
    for row in old_character_rows:
        if row["report_class"] == "review_clue":
            # A schema-adoption upgrade must not rewrite an already-classed
            # current clue or replace its explicit promotion outcome.
            continue
        prior_extra = row["extra"] if isinstance(row["extra"], dict) else {}
        if prior_extra.get("final_outcome") == "conflict":
            continue
        if prior_extra.get("final_outcome") in {
            "needs_confirmation", "unverifiable"
        }:
            bind.execute(
                sa.update(issues)
                .where(issues.c.id == row["id"])
                .values(report_class="review_clue")
            )
            continue
        if _historical_formal_character_issue(row):
            continue
        extra = {
            **prior_extra,
            "final_outcome": "unverifiable",
            "review_reason": "legacy_report_reclassified",
            "legacy_report_reclassified": True,
        }
        bind.execute(
            sa.update(issues)
            .where(issues.c.id == row["id"])
            .values(report_class="review_clue", extra=extra)
        )

    # Comparison rows are derived output.  A ready comparison that once
    # counted a now-demoted row must be regenerated under the new boundary.
    if {"analysis_run_comparisons", "issue_comparison_items"} <= table_names:
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
        demoted_runs = sa.select(issues.c.run_id).where(
            issues.c.report_class == "review_clue"
        )
        affected_comparisons = sa.select(comparisons.c.id).where(
            sa.or_(
                comparisons.c.baseline_run_id.in_(demoted_runs),
                comparisons.c.target_run_id.in_(demoted_runs),
            )
        )
        bind.execute(sa.delete(comparison_items).where(
            comparison_items.c.comparison_id.in_(affected_comparisons)
        ))
        bind.execute(
            sa.update(comparisons)
            .where(comparisons.c.id.in_(affected_comparisons))
            .values(status="pending", summary={}, provenance={}, completed_at=None)
        )
    if "ix_issues_run_report_class_id" not in {
        row["name"] for row in sa.inspect(bind).get_indexes("issues")
    }:
        op.create_index(
            "ix_issues_run_report_class_id",
            "issues", ["run_id", "report_class", "id"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    if "issues" not in sa.inspect(bind).get_table_names():
        return
    if "ix_issues_run_report_class_id" in {
        row["name"] for row in sa.inspect(bind).get_indexes("issues")
    }:
        op.drop_index("ix_issues_run_report_class_id", table_name="issues")
    if "report_class" in {
        row["name"] for row in sa.inspect(bind).get_columns("issues")
    }:
        op.drop_column("issues", "report_class")
