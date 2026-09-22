from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Thread
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import IntegrityError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Scope

from .config import get_settings
from .auth import (
    AuthContext,
    ExactOriginMiddleware,
    get_auth_context,
    require_csrf,
    router as auth_router,
)
from .db import (
    AnalysisDiagnosticRow,
    AnalysisRecordRow,
    AnalysisRunCharacterTraitInputRow,
    AnalysisRunComparisonRow,
    AnalysisRunExecutionRow,
    AnalysisRunInputContextRow,
    AnalysisRunInputNarrativeContextRow,
    AnalysisRunInputRow,
    AnalysisRunRow,
    CharacterTraitCandidateRow,
    CharacterTraitReviewRow,
    DocumentContextRow,
    DocumentNarrativeContextRevisionRow,
    DocumentRow,
    FeedbackRow,
    IssueComparisonItemRow,
    IssueRow,
    ProjectRow,
    RunEventRow,
    SessionLocal,
    init_db,
)
from .character_traits import (
    normalize_character_key,
    validate_character_trait_supersession,
)
from .character_trait_extraction import trait_keys_compatible
from .document_diff import build_document_diff
from .docx_import import DocxImportError, extract_docx_text
from .domain import CertaintyLevel, DocumentRole, EvidenceSpan, GraphResponse, SemanticModality, SourceScope, TimelineResponse
from .evaluation import run_evaluation
from .observability import AnalysisMetricsUnavailable, render_analysis_metrics
from .narrative_context import (
    NarrativeContextInput,
    NarrativeContextRevisionConflict,
    NarrativeContextRevisionInput,
    NarrativeScopeV1,
    add_context_revision,
    context_snapshot_payload,
    latest_context_revisions,
    payload_sha256,
    scope_relation,
)
from .narrative_context_inference import (
    MAX_INFERENCE_INPUT_CHARS,
    NarrativeContextInferenceInputError,
    NarrativeContextInferenceOutputError,
    infer_narrative_context,
)
from .projections import project_graph, project_timeline, record_sort_key
from .provider import (
    OpenAICompatibleProvider,
    ProviderError,
    RetryPolicy,
    safe_thinking_configuration,
    sanitize_request_id,
)
from .rate_limit import SlidingWindowLimiter, WriteRateLimitMiddleware
from .runtime_provenance import safe_runtime_provenance
from .run_comparison import (
    MATCHER_VERSION,
    build_input_diff,
    mark_comparison_unverifiable,
    materialize_run_comparison,
)
from .service import (
    DEFAULT_DOCUMENT_ROLE,
    DEFAULT_STORY_SCOPE,
    DISPATCH_FAILED_ERROR,
    MISSING_SNAPSHOT_ERROR,
    capture_run_inputs,
    CharacterTraitSnapshotLimitExceeded,
    copy_run_inputs,
    current_project_source_signature,
    document_content_sha256,
    execute_analysis,
    run_input_metadata,
    run_narrative_context_fingerprint,
    run_trait_snapshot_metadata,
    frozen_run_source_signature,
    safe_persisted_analysis_error,
)
from .time_utils import utc_now_naive

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="LoreGuard API", version="0.1.0", lifespan=lifespan)
settings = get_settings()
write_limiter = SlidingWindowLimiter(settings.rate_limit_per_minute, settings.rate_limit_window_seconds)
app.add_middleware(WriteRateLimitMiddleware, limiter=write_limiter)
app.add_middleware(
    ExactOriginMiddleware,
    allowed_origins=settings.parsed_cors_origins(),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parsed_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Accept", "Content-Type", "X-CSRF-Token", "Idempotency-Key"],
    expose_headers=["X-CSRF-Token"],
)
app.include_router(auth_router)


class SpaStaticFiles(StaticFiles):
    """Serve known client routes from ``index.html`` without hiding 404s.

    ``StaticFiles(html=True)`` only falls back to an index file for real
    directories. LoreGuard's browser routes are virtual, so a direct refresh
    of ``/login`` or ``/app/projects/...`` otherwise returns 404. The fallback
    is intentionally allow-listed: API paths, asset paths, file-like paths,
    unknown top-level paths, and traversal attempts retain normal static-file
    404 behaviour.
    """

    _LEGACY_WORKSPACE_ROUTES = frozenset(
        {"check", "projects", "diff", "visual", "audit", "report", "provider"}
    )

    @classmethod
    def _is_client_route(cls, path: str, request_path: str = "") -> bool:
        # Starlette builds nested static paths with the host OS separator, so
        # normalise Windows paths before applying URL-segment rules. Encoded
        # backslash traversal still becomes a ``..`` segment and is rejected.
        normalized_path = path.replace("\\", "/")
        normalized_request_path = request_path.replace("\\", "/")
        request_segments = [
            segment
            for segment in normalized_request_path.strip("/").split("/")
            if segment
        ]
        if any(segment in {".", ".."} for segment in request_segments):
            return False
        segments = [
            segment for segment in normalized_path.strip("/").split("/") if segment
        ]
        if any(segment in {".", ".."} for segment in segments):
            return False
        if not segments:
            return True
        if "." in segments[-1]:
            return False
        if len(segments) == 1:
            return segments[0] in {"login", "register", "app"} | cls._LEGACY_WORKSPACE_ROUTES
        return segments[0] == "app"

    async def get_response(self, path: str, scope: Scope) -> Response:
        not_found_response: Response | None = None
        try:
            response = await super().get_response(path, scope)
            if response.status_code != 404:
                return response
            not_found_response = response
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise

        if not self._is_client_route(path, str(scope.get("path", ""))):
            if not_found_response is not None:
                return not_found_response
            raise StarletteHTTPException(status_code=404)
        return await super().get_response("index.html", scope)


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""


class FeedbackIn(BaseModel):
    label: str
    comment: str = ""


class TextDocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)
    replace_document_id: str | None = None
    document_role: DocumentRole | None = None
    story_scope: str | None = Field(
        default=None,
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9_\-\u4e00-\u9fff]+$",
    )
    narrative_context: NarrativeContextInput | None = None


class CharacterTraitDecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["confirm", "reject"]
    expected_revision: int = Field(ge=0)
    comment: str = Field(default="", max_length=2_000)


class AnalysisRunIn(BaseModel):
    """Optional review-batch selector; an omitted body keeps legacy behavior."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["draft_review", "baseline_build", "full_review"] = (
        "full_review"
    )
    target_document_ids: list[str] | None = Field(
        default=None, max_length=256
    )
    sensitivity: Literal[
        "conservative", "balanced", "exploratory"
    ] = "balanced"


class NarrativeContextInferenceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=0)


ClarificationCategory = Literal[
    "scope_unknown",
    "missing_causal_bridge",
    "missing_state_transition",
    "ambiguous_reference",
    "insufficient_evidence",
]
ComparisonOutcome = Literal[
    "no_longer_detected", "persisting", "new", "unverifiable"
]


class SemanticReviewItemOut(BaseModel):
    id: str
    kind: Literal["clarification", "open_question"]
    category: ClarificationCategory | None = None
    modality: SemanticModality
    source_scope: SourceScope
    certainty: CertaintyLevel
    text: str
    evidence: EvidenceSpan


def _project_in_workspace(db, project_id: str, workspace_id: str) -> ProjectRow | None:
    return db.scalar(
        select(ProjectRow).where(
            ProjectRow.id == project_id,
            ProjectRow.workspace_id == workspace_id,
        )
    )


def _run_in_workspace(db, run_id: str, workspace_id: str) -> AnalysisRunRow | None:
    return db.scalar(
        select(AnalysisRunRow)
        .join(ProjectRow, ProjectRow.id == AnalysisRunRow.project_id)
        .where(
            AnalysisRunRow.id == run_id,
            ProjectRow.workspace_id == workspace_id,
        )
    )


def _review_batch_intent(payload: AnalysisRunIn) -> dict:
    requested = payload.target_document_ids
    if requested is not None:
        normalized = [value.strip() for value in requested]
        if any(not value or len(value) > 36 for value in normalized):
            raise HTTPException(422, "target_document_ids 包含无效文档 ID")
        if len(set(normalized)) != len(normalized):
            raise HTTPException(422, "target_document_ids 不能重复")
        requested = sorted(normalized)
    if payload.mode != "draft_review" and requested:
        raise HTTPException(
            422, "target_document_ids 仅适用于 draft_review 模式"
        )
    if payload.mode == "draft_review" and requested == []:
        raise HTTPException(
            422, "draft_review 的显式 target_document_ids 不能为空"
        )
    return {
        "mode": payload.mode,
        "sensitivity": payload.sensitivity,
        "requested_target_document_ids": requested,
    }


def _review_batch_selection(
    db,
    *,
    project_id: str,
    payload: AnalysisRunIn,
) -> tuple[list[DocumentRow], dict[str, str], dict]:
    """Select a closed target/background set from server-owned metadata."""

    intent = _review_batch_intent(payload)
    documents = list(
        db.scalars(
            select(DocumentRow)
            .where(
                DocumentRow.project_id == project_id,
                DocumentRow.active.is_(True),
            )
            .order_by(DocumentRow.created_at, DocumentRow.id)
        ).all()
    )
    if not documents:
        raise HTTPException(400, "项目没有可分析文档")

    if payload.mode == "full_review":
        roles = {row.id: "target" for row in documents}
        coverage = {
            **intent,
            "selected_document_ids": [row.id for row in documents],
            "target_document_ids": [row.id for row in documents],
            "background_document_ids": [],
            "excluded_documents": [],
        }
        return documents, roles, coverage

    contexts = {
        row.document_id: row
        for row in db.scalars(
            select(DocumentContextRow).where(
                DocumentContextRow.document_id.in_([row.id for row in documents])
            )
        ).all()
    }
    revisions = latest_context_revisions(db, [row.id for row in documents])
    document_by_id = {row.id: row for row in documents}
    metadata: dict[str, tuple[str, dict]] = {}
    for document in documents:
        legacy = contexts.get(document.id)
        role = legacy.document_role if legacy else DEFAULT_DOCUMENT_ROLE
        story_scope = legacy.story_scope if legacy else DEFAULT_STORY_SCOPE
        metadata[document.id] = (
            role,
            context_snapshot_payload(
                revisions.get(document.id),
                document_role=role,
                story_scope=story_scope,
            ),
        )

    excluded: list[dict[str, str]] = []
    excluded_ids: set[str] = set()
    target_selection_exclusions: list[dict[str, str]] = []

    def exclude(document: DocumentRow, reason: str) -> None:
        if document.id in excluded_ids:
            return
        excluded_ids.add(document.id)
        excluded.append(
            {
                "document_id": document.id,
                "document_name": document.name,
                "reason": reason,
            }
        )

    targets: list[DocumentRow] = []
    if payload.mode == "draft_review":
        requested = intent["requested_target_document_ids"]
        if requested is not None:
            # The same generic 404 covers foreign-workspace, foreign-project,
            # inactive, and unknown identifiers without disclosing existence.
            if any(document_id not in document_by_id for document_id in requested):
                raise HTTPException(404, "目标文档不存在")
            candidates = [document_by_id[document_id] for document_id in requested]
            blocking: list[dict[str, str]] = []
            for document in candidates:
                role, narrative = metadata[document.id]
                if role != "chapter":
                    blocking.append(
                        {
                            "document_id": document.id,
                            "document_name": document.name,
                            "reason": "target_must_be_chapter",
                        }
                    )
                elif narrative.get("resolution_state") != "confirmed":
                    blocking.append(
                        {
                            "document_id": document.id,
                            "document_name": document.name,
                            "reason": "narrative_context_unconfirmed",
                        }
                    )
                elif narrative.get("publication_status") not in {
                    "draft",
                    "in_review",
                }:
                    blocking.append(
                        {
                            "document_id": document.id,
                            "document_name": document.name,
                            "reason": "not_draft_or_in_review",
                        }
                    )
                else:
                    targets.append(document)
            if blocking:
                raise HTTPException(
                    409,
                    detail={
                        "code": "invalid_draft_review_targets",
                        "message": "显式目标必须是已确认的草稿或审阅中文档",
                        "blocking_documents": blocking,
                    },
                )
        else:
            for document in documents:
                role, narrative = metadata[document.id]
                publication = narrative.get("publication_status")
                if publication not in {"draft", "in_review"}:
                    continue
                if role != "chapter":
                    target_selection_exclusions.append(
                        {
                            "document_id": document.id,
                            "document_name": document.name,
                            "reason": "target_must_be_chapter",
                        }
                    )
                    continue
                if narrative.get("resolution_state") != "confirmed":
                    exclude(document, "narrative_context_unconfirmed")
                    continue
                targets.append(document)
        if not targets:
            raise HTTPException(
                409,
                detail={
                    "code": "no_eligible_draft_targets",
                    "message": "没有已确认的草稿或审阅中文档可作为审查目标",
                    "excluded_documents": [
                        *excluded,
                        *target_selection_exclusions,
                    ],
                },
            )

    target_ids = {row.id for row in targets}
    backgrounds: list[DocumentRow] = []
    for document in documents:
        if document.id in target_ids:
            continue
        role, narrative = metadata[document.id]
        if narrative.get("resolution_state") != "confirmed":
            exclude(document, "narrative_context_unconfirmed")
            continue
        if narrative.get("publication_status") == "retired":
            exclude(document, "retired_document")
            continue
        if (
            payload.mode == "baseline_build"
            and narrative.get("publication_status") in {"draft", "in_review"}
        ):
            exclude(document, "draft_excluded_from_baseline")
            continue
        is_authority = role in {"canon", "character_profile"}
        is_published_history = (
            narrative.get("publication_status") == "published"
        )
        if not (is_authority or is_published_history):
            exclude(document, "not_authority_or_published_history")
            continue
        if targets and not any(
            scope_relation(
                narrative.get("scope", {}),
                metadata[target.id][1].get("scope", {}),
                first_resolution=str(narrative.get("resolution_state", "")),
                second_resolution=str(
                    metadata[target.id][1].get("resolution_state", "")
                ),
            )
            == "compatible"
            for target in targets
        ):
            exclude(document, "narrative_scope_incompatible")
            continue
        backgrounds.append(document)

    selected = [
        row for row in documents if row.id in target_ids or row in backgrounds
    ]
    if payload.mode == "baseline_build" and not selected:
        raise HTTPException(
            409,
            detail={
                "code": "no_eligible_baseline_inputs",
                "message": "没有已确认的设定、角色资料或已发布历史可建立基线",
                "excluded_documents": excluded,
            },
        )
    roles = {
        row.id: ("target" if row.id in target_ids else "background")
        for row in selected
    }
    coverage = {
        **intent,
        "selected_document_ids": [row.id for row in selected],
        "target_document_ids": [row.id for row in selected if roles[row.id] == "target"],
        "background_document_ids": [
            row.id for row in selected if roles[row.id] == "background"
        ],
        "excluded_documents": excluded,
        "target_selection_exclusions": target_selection_exclusions,
    }
    return selected, roles, coverage


def _issue_in_workspace(db, issue_id: str, workspace_id: str) -> IssueRow | None:
    return db.scalar(
        select(IssueRow)
        .join(AnalysisRunRow, AnalysisRunRow.id == IssueRow.run_id)
        .join(ProjectRow, ProjectRow.id == AnalysisRunRow.project_id)
        .where(
            IssueRow.id == issue_id,
            ProjectRow.workspace_id == workspace_id,
        )
    )


def serialize_run(row: AnalysisRunRow, db=None) -> dict:
    payload = {key: getattr(row, key) for key in ("id", "project_id", "status", "created_at", "started_at", "completed_at", "input_chars", "prompt_tokens", "completion_tokens", "error")}
    payload["error"] = safe_persisted_analysis_error(row.error)
    prices_configured = settings.model_input_price_per_million is not None and settings.model_output_price_per_million is not None
    payload["estimated_cost_usd"] = row.estimated_cost_usd if prices_configured else None
    coverage = row.batch_coverage if isinstance(row.batch_coverage, dict) else {}
    payload["review_batch"] = {
        "mode": row.batch_mode or "full_review",
        "sensitivity": row.sensitivity or "balanced",
        "target_document_ids": list(coverage.get("target_document_ids", [])),
        "background_document_ids": list(
            coverage.get("background_document_ids", [])
        ),
        "excluded_documents": list(coverage.get("excluded_documents", [])),
        "target_selection_exclusions": list(
            coverage.get("target_selection_exclusions", [])
        ),
    }
    if db is not None:
        payload["input_documents"] = run_input_metadata(db, row.id)
        if not coverage and payload["input_documents"]:
            payload["review_batch"]["target_document_ids"] = [
                item["document_id"]
                for item in payload["input_documents"]
                if item.get("batch_role", "target") == "target"
            ]
            payload["review_batch"]["background_document_ids"] = [
                item["document_id"]
                for item in payload["input_documents"]
                if item.get("batch_role") == "background"
            ]
        diagnostic = db.get(AnalysisDiagnosticRow, row.id)
        usage_accounting = (
            diagnostic.payload.get("usage_accounting")
            if diagnostic and isinstance(diagnostic.payload, dict)
            else None
        )
        # Additive API qualifier: callers must not interpret interrupted
        # Agent-only token counters as a complete analysis-run total.
        payload["usage_accounting"] = usage_accounting
        execution = db.get(AnalysisRunExecutionRow, row.id)
        payload["attempt_no"] = execution.attempt_no if execution else None
        payload["retried_from"] = execution.retried_from_run_id if execution else None
        comparison = db.scalar(
            select(AnalysisRunComparisonRow).where(
                AnalysisRunComparisonRow.target_run_id == row.id
            )
        )
        payload["comparison_id"] = comparison.id if comparison else None
        payload["comparison_baseline_run_id"] = (
            comparison.baseline_run_id if comparison else None
        )
        payload["input_snapshot_available"] = bool(payload["input_documents"])
        payload["narrative_context_snapshot_sha256"] = (
            run_narrative_context_fingerprint(db, row.id)
            if payload["input_documents"]
            else None
        )
        payload.update(run_trait_snapshot_metadata(db, row.id))
        if row.status == "completed":
            counts = dict(
                db.execute(
                    select(AnalysisRecordRow.kind, func.count())
                    .where(
                        AnalysisRecordRow.run_id == row.id,
                        AnalysisRecordRow.kind.in_(("clarification", "open_question")),
                    )
                    .group_by(AnalysisRecordRow.kind)
                ).all()
            )
            payload["clarification_count"] = counts.get("clarification", 0)
            payload["open_question_count"] = counts.get("open_question", 0)
        else:
            payload["clarification_count"] = None
            payload["open_question_count"] = None
    return payload


def serialize_document(row: DocumentRow, include_content: bool = True, db=None) -> dict:
    context = db.get(DocumentContextRow, row.id) if db is not None else None
    narrative = (
        latest_context_revisions(db, [row.id]).get(row.id)
        if db is not None
        else None
    )
    document_role = context.document_role if context else DEFAULT_DOCUMENT_ROLE
    story_scope = context.story_scope if context else DEFAULT_STORY_SCOPE
    payload = {
        "id": row.id,
        "project_id": row.project_id,
        "name": row.name,
        "version": row.version,
        "active": row.active,
        "created_at": row.created_at,
        "document_role": document_role,
        "story_scope": story_scope,
        "context_explicit": context is not None,
        "narrative_context": context_snapshot_payload(
            narrative,
            document_role=document_role,
            story_scope=story_scope,
        ),
    }
    if include_content:
        payload["content"] = row.content
    return payload


def serialize_narrative_context_revision(
    row: DocumentNarrativeContextRevisionRow,
) -> dict:
    payload = {
        "id": row.id,
        "project_id": row.project_id,
        "document_id": row.document_id,
        "revision": row.revision,
        "resolution_state": row.resolution_state,
        "origin": row.origin,
        "authority_tier": row.authority_tier,
        "publication_status": row.publication_status,
        "scope": row.scope_payload,
        "scope_sha256": row.scope_sha256,
        "inference_confidence": row.inference_confidence,
        "created_at": row.created_at,
    }
    if row.origin == "model_inferred":
        payload["inference"] = {
            "confidence": row.inference_confidence,
            "reasoning": row.inference_reasoning or "",
            "evidence": (
                row.inference_evidence
                if isinstance(row.inference_evidence, list)
                else []
            ),
            "usage": (
                row.inference_usage
                if isinstance(row.inference_usage, dict)
                else {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            ),
        }
    else:
        payload["inference"] = None
    return payload


def serialize_character_trait_candidate(row: CharacterTraitCandidateRow) -> dict:
    return {
        "id": row.id,
        "project_id": row.project_id,
        "source_run_id": row.source_run_id,
        "character_key": row.character_key,
        "character_display_name": row.character_display_name,
        "trait_type": row.trait_type,
        "trait_key": row.trait_key,
        "value": row.value,
        "polarity": row.polarity,
        "stability": row.stability,
        "contexts": row.contexts,
        "origin": row.origin,
        "authority_tier": row.authority_tier,
        "confidence": row.confidence,
        "scope": row.scope_payload,
        "scope_sha256": row.scope_sha256,
        "valid_from_release_ordinal": row.valid_from_release_ordinal,
        "valid_until_release_ordinal": row.valid_until_release_ordinal,
        "evidence": row.evidence,
        "candidate_fingerprint": row.candidate_fingerprint,
        "generator_version": row.generator_version,
        "review_state": row.review_state,
        "revision": row.lock_version,
        "supersedes_candidate_id": row.supersedes_candidate_id,
        "reviewed_at": row.reviewed_at,
        "created_at": row.created_at,
    }


def _candidate_source_is_current(
    db, row: CharacterTraitCandidateRow
) -> tuple[bool, str | None]:
    source_run = db.get(AnalysisRunRow, row.source_run_id)
    if (
        source_run is None
        or source_run.project_id != row.project_id
        or source_run.status != "completed"
    ):
        return False, "来源分析尚未完整完成"
    evidence = row.evidence if isinstance(row.evidence, list) else []
    bindings: dict[str, tuple[str, int, str]] = {}
    for item in evidence:
        if not isinstance(item, dict):
            return False, "候选缺少可核对的来源文档"
        input_id = item.get("input_id")
        document_id = item.get("document_id")
        document_version = item.get("document_version")
        content_sha256 = item.get("content_sha256")
        if (
            not isinstance(input_id, str)
            or not input_id
            or not isinstance(document_id, str)
            or not document_id
            or type(document_version) is not int
            or document_version < 1
            or not isinstance(content_sha256, str)
            or len(content_sha256) != 64
        ):
            return False, "候选缺少可核对的来源文档"
        binding = (document_id, document_version, content_sha256)
        if input_id in bindings and bindings[input_id] != binding:
            return False, "候选来源快照存在歧义"
        bindings[input_id] = binding
    if not bindings:
        return False, "候选缺少可核对的来源文档"

    frozen_inputs = list(
        db.scalars(
            select(AnalysisRunInputRow).where(
                AnalysisRunInputRow.run_id == row.source_run_id,
                AnalysisRunInputRow.id.in_(bindings),
            )
        ).all()
    )
    frozen_by_id = {item.id: item for item in frozen_inputs}
    if len(frozen_by_id) != len(bindings):
        return False, "候选来源快照已缺失"
    for input_id, binding in bindings.items():
        frozen = frozen_by_id[input_id]
        if binding != (
            frozen.document_id,
            frozen.document_version,
            frozen.content_sha256,
        ):
            return False, "候选来源与冻结输入不一致"

    document_ids = {item.document_id for item in frozen_inputs}
    active_documents = list(
        db.scalars(
            select(DocumentRow).where(
                DocumentRow.project_id == row.project_id,
                DocumentRow.id.in_(document_ids),
                DocumentRow.active.is_(True),
            )
        ).all()
    )
    active_by_id = {item.id: item for item in active_documents}
    if len(active_by_id) != len(document_ids):
        return False, "来源文档已被新版本替代，请重新分析后确认"

    for frozen in frozen_inputs:
        current = active_by_id[frozen.document_id]
        if (
            current.version != frozen.document_version
            or document_content_sha256(current.content) != frozen.content_sha256
        ):
            return False, "来源文档已变更，请重新分析后确认"

    legacy_rows = list(
        db.scalars(
            select(DocumentContextRow).where(
                DocumentContextRow.document_id.in_(document_ids)
            )
        ).all()
    )
    legacy_by_document = {item.document_id: item for item in legacy_rows}
    if len(legacy_by_document) != len(document_ids):
        return False, "来源文档的叙事上下文已缺失"

    current_revisions = latest_context_revisions(db, list(document_ids))
    if set(current_revisions) != document_ids:
        return False, "来源文档的叙事上下文已缺失"

    frozen_context_rows = list(
        db.scalars(
            select(AnalysisRunInputNarrativeContextRow).where(
                AnalysisRunInputNarrativeContextRow.input_id.in_(bindings)
            )
        ).all()
    )
    frozen_context_by_input = {
        item.input_id: item for item in frozen_context_rows
    }
    if len(frozen_context_by_input) != len(bindings):
        return False, "来源分析的冻结叙事上下文已缺失"

    for frozen in frozen_inputs:
        frozen_context = frozen_context_by_input[frozen.id]
        if (
            frozen_context.schema_version != 1
            or not isinstance(frozen_context.payload, dict)
            or payload_sha256(frozen_context.payload)
            != frozen_context.payload_sha256
        ):
            return False, "来源分析的冻结叙事上下文无法校验"
        legacy = legacy_by_document[frozen.document_id]
        try:
            current_payload = context_snapshot_payload(
                current_revisions[frozen.document_id],
                document_role=legacy.document_role,
                story_scope=legacy.story_scope,
            )
        except (TypeError, ValueError):
            return False, "来源文档的叙事上下文无法校验"
        if (
            current_payload != frozen_context.payload
            or payload_sha256(current_payload) != frozen_context.payload_sha256
        ):
            return False, "来源文档的叙事上下文已变更，请重新分析后确认"
    return True, None


def serialize_character_trait_candidate_for_review(
    db, row: CharacterTraitCandidateRow
) -> dict:
    payload = serialize_character_trait_candidate(row)
    current, reason = _candidate_source_is_current(db, row)
    if row.review_state == "pending" and not current:
        payload.update(
            {
                "status": "stale",
                "reviewable": False,
                "unreviewable_reason": reason,
            }
        )
    else:
        payload.update(
            {
                "status": row.review_state,
                "reviewable": row.review_state == "pending",
                "unreviewable_reason": None,
            }
        )
    return payload


def resolve_document_context(
    db,
    same_name: list[DocumentRow],
    replace_document_id: str | None,
    document_role: DocumentRole | None,
    story_scope: str | None,
) -> tuple[str, str]:
    """Inherit context from the replaced/latest version unless overridden."""
    base = None
    if replace_document_id:
        base = next((row for row in same_name if row.id == replace_document_id), None)
    if base is None and same_name:
        base = max(same_name, key=lambda row: row.version)
    inherited = db.get(DocumentContextRow, base.id) if base is not None else None
    role = (
        document_role.value
        if document_role is not None
        else inherited.document_role
        if inherited is not None
        else DEFAULT_DOCUMENT_ROLE
    )
    scope = (
        story_scope
        if story_scope is not None
        else inherited.story_scope
        if inherited is not None
        else DEFAULT_STORY_SCOPE
    )
    return role, scope


def dispatch_analysis(run_id: str) -> None:
    if settings.use_celery:
        from .tasks import analyze_project

        analyze_project.delay(run_id)
    else:
        Thread(target=execute_analysis, args=(run_id,), daemon=True).start()


def _normalize_idempotency_key(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise HTTPException(400, "Idempotency-Key 不能为空")
    if len(normalized) > 128 or re.fullmatch(r"[A-Za-z0-9._:-]+", normalized) is None:
        raise HTTPException(
            400,
            "Idempotency-Key 仅支持 1–128 位字母、数字及 . _ : -",
        )
    return normalized


def _idempotent_run(
    db, project_id: str, idempotency_key: str | None
) -> AnalysisRunRow | None:
    if idempotency_key is None:
        return None
    return db.scalar(
        select(AnalysisRunRow).where(
            AnalysisRunRow.project_id == project_id,
            AnalysisRunRow.idempotency_key == idempotency_key,
        )
    )


def _create_run_or_load_winner(
    db,
    run: AnalysisRunRow,
    idempotency_key: str | None,
    prepare: Callable[[AnalysisRunRow], object],
) -> tuple[AnalysisRunRow, bool]:
    """Create a run and snapshot, or load the concurrent winner for its key.

    The pre-insert lookup is only a fast path.  Correctness comes from the
    database unique index and this IntegrityError recovery path.
    """
    project_id = run.project_id
    try:
        db.add(run)
        db.flush()
        prepare(run)
        db.commit()
        return run, True
    except IntegrityError:
        db.rollback()
        winner = _idempotent_run(db, project_id, idempotency_key)
        if winner is None:
            raise
        return winner, False


def _accepted_run_payload(db, run: AnalysisRunRow, *, created: bool) -> dict:
    execution = db.get(AnalysisRunExecutionRow, run.id)
    coverage = run.batch_coverage if isinstance(run.batch_coverage, dict) else {}
    if coverage:
        target_ids = list(coverage.get("target_document_ids", []))
        background_ids = list(coverage.get("background_document_ids", []))
    else:
        frozen = run_input_metadata(db, run.id)
        target_ids = [
            item["document_id"]
            for item in frozen
            if item.get("batch_role", "target") == "target"
        ]
        background_ids = [
            item["document_id"]
            for item in frozen
            if item.get("batch_role") == "background"
        ]
    return {
        "id": run.id,
        "project_id": run.project_id,
        "status": run.status,
        "retried_from": execution.retried_from_run_id if execution else None,
        "deduplicated": not created,
        "review_batch": {
            "mode": run.batch_mode or "full_review",
            "sensitivity": run.sensitivity or "balanced",
            "target_document_ids": target_ids,
            "background_document_ids": background_ids,
            "excluded_documents": list(
                coverage.get("excluded_documents", [])
            ),
            "target_selection_exclusions": list(
                coverage.get("target_selection_exclusions", [])
            ),
        },
    }


def _require_idempotency_operation(
    db,
    run: AnalysisRunRow,
    *,
    retried_from_run_id: str | None,
    recheck_baseline_run_id: str | None = None,
    review_batch_intent: dict | None = None,
) -> None:
    """Reject reuse of a project-scoped key for a different user intent."""
    execution = db.get(AnalysisRunExecutionRow, run.id)
    actual_source = execution.retried_from_run_id if execution else None
    comparison = db.scalar(
        select(AnalysisRunComparisonRow).where(
            AnalysisRunComparisonRow.target_run_id == run.id
        )
    )
    actual_baseline = comparison.baseline_run_id if comparison else None
    coverage = run.batch_coverage if isinstance(run.batch_coverage, dict) else {}
    actual_intent = {
        "mode": run.batch_mode or "full_review",
        "sensitivity": run.sensitivity or "balanced",
        "requested_target_document_ids": coverage.get(
            "requested_target_document_ids"
        ),
    }
    if (
        actual_source == retried_from_run_id
        and actual_baseline == recheck_baseline_run_id
        and (
            review_batch_intent is None
            or actual_intent == review_batch_intent
        )
    ):
        return
    raise HTTPException(
        409,
        detail={
            "code": "idempotency_key_conflict",
            "message": "这个 Idempotency-Key 已用于另一项分析操作，请为新操作生成新键",
        },
    )


def _mark_dispatch_failure(run_id: str) -> bool:
    """Persist a safe terminal state if scheduling fails before worker claim."""
    with SessionLocal() as db:
        changed = db.execute(
            update(AnalysisRunRow)
            .where(
                AnalysisRunRow.id == run_id,
                AnalysisRunRow.status == "queued",
            )
            .values(
                status="failed",
                error=DISPATCH_FAILED_ERROR,
                completed_at=utc_now_naive(),
            )
        ).rowcount
        if changed != 1:
            db.rollback()
            return False
        db.add(
            RunEventRow(
                run_id=run_id,
                stage="failed",
                progress=100,
                message="任务调度失败，输入快照已保留，可稍后重试",
            )
        )
        db.commit()
        return True


def _dispatch_created_run(run_id: str) -> None:
    try:
        dispatch_analysis(run_id)
    except Exception:
        marked_failed = _mark_dispatch_failure(run_id)
        if marked_failed:
            raise HTTPException(
                503,
                detail={
                    "code": "analysis_dispatch_failed",
                    "message": "任务暂未进入执行队列，请稍后重试",
                    "run_id": run_id,
                    "retryable": True,
                },
                headers={"Retry-After": "1"},
            ) from None
        # A worker may have claimed the run in the narrow interval after a
        # transport-ambiguous dispatch.  Do not overwrite that durable state.


def enforce_daily_model_budget(db, workspace_id: str | None = None) -> None:
    """Reject model-backed work after the local daily usage threshold.

    This is intentionally a single-database check, not a distributed quota
    reservation.  The per-run gate remains the hard fallback for concurrent
    local jobs.
    """
    model_requested = (
        settings.enable_model_extraction
        or settings.enable_evidence_investigator
        or settings.enable_issue_evidence_review
        or settings.enable_character_consistency
    ) and bool(settings.openai_api_key.strip())
    if not model_requested:
        return
    now = utc_now_naive()
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    statement = (
        select(
            AnalysisRunRow.prompt_tokens,
            AnalysisRunRow.completion_tokens,
            AnalysisDiagnosticRow.payload,
        )
        .join(ProjectRow, ProjectRow.id == AnalysisRunRow.project_id)
        .outerjoin(
            AnalysisDiagnosticRow,
            AnalysisDiagnosticRow.run_id == AnalysisRunRow.id,
        )
        .where(
            AnalysisRunRow.created_at >= day_start,
            # Cancelled runs can contain an explicitly qualified lower-bound
            # Agent usage record. Those consumed tokens still count against
            # the local daily safety budget.
            AnalysisRunRow.status.in_(
                ("queued", "running", "completed", "failed", "cancelled")
            ),
        )
    )
    if workspace_id is not None:
        statement = statement.where(ProjectRow.workspace_id == workspace_id)
    usage_rows = db.execute(statement).all()
    daily_usage = sum(_conservative_run_token_debit(*row) for row in usage_rows)
    if settings.daily_token_budget <= 0 or daily_usage >= settings.daily_token_budget:
        seconds_to_reset = max(
            1, int(86400 - (now - day_start).total_seconds())
        )
        raise HTTPException(
            429,
            f"当日模型 Token 预算已用尽（{daily_usage}/{settings.daily_token_budget}）",
            headers={"Retry-After": str(seconds_to_reset)},
        )


def _conservative_run_token_debit(
    prompt_tokens: object,
    completion_tokens: object,
    diagnostic_payload: object,
) -> int:
    """Use reported usage plus any proven conservative review debit delta."""
    reported = sum(
        value if type(value) is int and value >= 0 else 0
        for value in (prompt_tokens, completion_tokens)
    )
    if not isinstance(diagnostic_payload, dict):
        return reported
    accounting = diagnostic_payload.get("usage_accounting")
    if not isinstance(accounting, dict):
        review = diagnostic_payload.get("ai_evidence_review")
        if isinstance(review, dict):
            nested = review.get("usage_accounting")
            accounting = nested if isinstance(nested, dict) else review
    if not isinstance(accounting, dict):
        return reported
    charged = accounting.get("charged_tokens")
    review_reported = sum(
        value if type(value) is int and value >= 0 else 0
        for value in (
            accounting.get("prompt_tokens"),
            accounting.get("completion_tokens"),
        )
    )
    if type(charged) is not int or charged < review_reported:
        return reported
    return reported + charged - review_reported


def prepare_document_version(
    db,
    project_id: str,
    name: str,
    replace_document_id: str | None,
    document_role: DocumentRole | None = None,
    story_scope: str | None = None,
) -> tuple[int, list[str], str, str]:
    # Serialize version allocation per project on PostgreSQL. The unique
    # logical-name/version and partial-active indexes remain authoritative and
    # also protect SQLite/tests where SELECT FOR UPDATE is advisory/no-op.
    locked_project = db.scalar(
        select(ProjectRow).where(ProjectRow.id == project_id).with_for_update()
    )
    if locked_project is None:
        raise HTTPException(404, "项目不存在")
    same_name = db.scalars(
        select(DocumentRow).where(
            DocumentRow.project_id == project_id,
            func.lower(DocumentRow.name) == name.lower(),
        )
    ).all()
    if replace_document_id:
        old = db.scalar(
            select(DocumentRow).where(
                DocumentRow.id == replace_document_id,
                DocumentRow.project_id == project_id,
            )
        )
        if not old:
            raise HTTPException(404, "待替换文档不存在")
        if old.name.lower() != name.lower():
            raise HTTPException(409, "替换文档必须保持同名；如需新文件请直接上传")
    resolved_role, resolved_scope = resolve_document_context(
        db,
        list(same_name),
        replace_document_id,
        document_role,
        story_scope,
    )
    version = max((row.version for row in same_name), default=0) + 1
    superseded = []
    for row in same_name:
        if row.active:
            row.active = False
            superseded.append(row.id)
    return version, superseded, resolved_role, resolved_scope


def add_initial_narrative_context(
    db,
    *,
    project_id: str,
    document_id: str,
    document_role: str,
    narrative_context: NarrativeContextInput | None,
    user_id: str | None,
) -> DocumentNarrativeContextRevisionRow:
    supplied = narrative_context or NarrativeContextInput()
    return add_context_revision(
        db,
        project_id=project_id,
        document_id=document_id,
        document_role=document_role,
        resolution_state=supplied.resolution_state,
        publication_status=supplied.publication_status,
        scope=supplied.scope,
        origin="explicit",
        created_by_user_id=user_id,
        expected_revision=0,
    )


def parse_multipart_narrative_context(
    value: str | None,
) -> NarrativeContextInput | None:
    if value is None or not value.strip():
        return None
    try:
        return NarrativeContextInput.model_validate_json(value)
    except Exception:
        raise HTTPException(
            422,
            detail={
                "code": "invalid_narrative_context",
                "message": "叙事上下文必须是符合 schema v1 的 JSON",
            },
        ) from None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "time": utc_now_naive().isoformat(),
        "model": {
            "configured": (
                settings.enable_model_extraction
                or settings.enable_evidence_investigator
                or settings.enable_issue_evidence_review
                or settings.enable_character_consistency
            )
            and bool(settings.openai_api_key.strip()),
            "thinking": safe_thinking_configuration(settings),
        },
        "runtime_provenance": safe_runtime_provenance(settings),
    }


def _provider_check_suggestions(category: str) -> list[str]:
    if category == "not_configured":
        return [
            "启用模型抽取、证据调查器或 AI 证据复核，并配置有效的 API 凭据后重试。"
        ]
    if category == "unauthorized":
        return [
            "检查 API 凭据是否有效、未过期，并确认凭据属于当前租户或项目。",
            "确认凭据已获得模型调用授权。",
        ]
    if category == "forbidden":
        return [
            "确认账户额度与计费状态可用。",
            "确认当前租户或项目有权访问所选模型。",
            "检查服务方的 IP 白名单、区域或网关访问策略。",
            "确认 API 凭据具有模型调用权限。",
        ]
    if category in {"rate_limit", "upstream_5xx"}:
        return ["服务已响应，请稍后重试；若持续失败，请检查额度和服务状态。"]
    if category in {"connect_timeout", "read_timeout", "transport"}:
        return ["检查到模型服务的网络连通性后重试。"]
    if category == "invalid_response":
        return ["模型服务已连通，但返回内容不符合最小 JSON 协议；请检查模型的 JSON 输出兼容性。"]
    if category == "success":
        return []
    return ["检查模型兼容性与访问权限后重试。"]


def _provider_check_metric(value: object) -> int | None:
    """Allowlist a bounded non-negative integer for the public preflight."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0 or value > 2_147_483_647:
        return None
    return value


def _provider_check_metrics(
    telemetry: object | None,
    *,
    fallback_elapsed_ms: object | None = None,
) -> dict[str, object]:
    elapsed_ms = _provider_check_metric(
        getattr(telemetry, "elapsed_ms", fallback_elapsed_ms)
    )
    prompt_tokens = _provider_check_metric(
        getattr(telemetry, "prompt_tokens", None)
    )
    completion_tokens = _provider_check_metric(
        getattr(telemetry, "completion_tokens", None)
    )
    return {
        "latency_ms": elapsed_ms,
        "token_usage": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": (
                prompt_tokens + completion_tokens
                if prompt_tokens is not None and completion_tokens is not None
                else None
            ),
        },
    }


@app.post("/api/v1/model/provider-check")
def check_model_provider(
    _context: AuthContext = Depends(require_csrf),
) -> dict:
    """Run an explicit, minimal provider preflight and return safe diagnostics."""
    provider = OpenAICompatibleProvider(settings)
    thinking = safe_thinking_configuration(settings)
    if not provider.configured:
        category = "not_configured"
        return {
            "status": category,
            "configured": False,
            "json_contract_ok": None,
            "reachable": False,
            "authorized": False,
            "category": category,
            "http_status": None,
            "request_id": None,
            "thinking": thinking,
            "suggestions": _provider_check_suggestions(category),
            **_provider_check_metrics(None),
        }

    try:
        result = provider.complete(
            "Return only valid JSON.",
            'Return {"status":"ok"}.',
        )
    except ProviderError as exc:
        category = exc.category
        http_status = exc.http_status
        reachable = http_status is not None
        authorized = False if category in {"unauthorized", "forbidden"} else None
        telemetry = getattr(exc, "telemetry", None)
        return {
            "status": category,
            "configured": True,
            "json_contract_ok": None,
            "reachable": reachable,
            "authorized": authorized,
            "category": category,
            "http_status": http_status,
            "request_id": sanitize_request_id(exc.request_id),
            "thinking": thinking,
            "suggestions": _provider_check_suggestions(category),
            **_provider_check_metrics(
                telemetry,
                fallback_elapsed_ms=getattr(exc, "elapsed_ms", None),
            ),
        }

    category = "success"
    try:
        contract_ok = json.loads(result.text) == {"status": "ok"}
    except (json.JSONDecodeError, TypeError):
        contract_ok = False
    if not contract_ok:
        category = "invalid_response"
    telemetry = getattr(result, "telemetry", None)
    return {
        "status": category,
        "configured": True,
        "json_contract_ok": contract_ok,
        "reachable": True,
        "authorized": True,
        "category": category,
        "http_status": getattr(telemetry, "http_status", 200),
        "request_id": sanitize_request_id(
            getattr(telemetry, "request_id", None)
        ),
        "thinking": thinking,
        "suggestions": _provider_check_suggestions(category),
        **_provider_check_metrics(telemetry),
    }


@app.post("/api/v1/projects", status_code=201)
def create_project(
    payload: ProjectIn,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    with SessionLocal() as db:
        row = ProjectRow(
            workspace_id=context.workspace_id,
            name=payload.name,
            description=payload.description,
        )
        db.add(row); db.commit()
        return {"id": row.id, "name": row.name, "description": row.description, "created_at": row.created_at}


@app.get("/api/v1/projects")
def list_projects(
    context: AuthContext = Depends(get_auth_context),
) -> list[dict]:
    with SessionLocal() as db:
        projects = db.scalars(
            select(ProjectRow)
            .where(ProjectRow.workspace_id == context.workspace_id)
            .order_by(ProjectRow.created_at.desc())
        ).all()
        result = []
        for project in projects:
            active_document_count = db.scalar(
                select(func.count()).select_from(DocumentRow).where(
                    DocumentRow.project_id == project.id, DocumentRow.active.is_(True)
                )
            ) or 0
            latest_run = db.scalar(
                select(AnalysisRunRow)
                .where(AnalysisRunRow.project_id == project.id)
                .order_by(AnalysisRunRow.created_at.desc())
                .limit(1)
            )
            result.append({
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "created_at": project.created_at,
                "active_document_count": active_document_count,
                "latest_run": serialize_run(latest_run, db) if latest_run else None,
            })
        return result


@app.get("/api/v1/projects/{project_id}")
def get_project(
    project_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    with SessionLocal() as db:
        project = _project_in_workspace(db, project_id, context.workspace_id)
        if not project:
            raise HTTPException(404, "项目不存在")
        documents = db.scalars(
            select(DocumentRow)
            .where(DocumentRow.project_id == project_id, DocumentRow.active.is_(True))
            .order_by(DocumentRow.created_at)
        ).all()
        return {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "documents": [serialize_document(d, db=db) for d in documents],
        }


@app.get("/api/v1/projects/{project_id}/documents")
def list_documents(
    project_id: str,
    include_history: bool = False,
    context: AuthContext = Depends(get_auth_context),
) -> list[dict]:
    with SessionLocal() as db:
        if not _project_in_workspace(db, project_id, context.workspace_id):
            raise HTTPException(404, "项目不存在")
        statement = select(DocumentRow).where(DocumentRow.project_id == project_id)
        if not include_history:
            statement = statement.where(DocumentRow.active.is_(True))
        rows = db.scalars(statement.order_by(DocumentRow.name, DocumentRow.version.desc())).all()
        # Version pickers and project history only need metadata. Returning every
        # historical body makes the browser download all revisions of a large
        # story before it can render the workspace.
        return [serialize_document(row, include_content=False, db=db) for row in rows]


@app.get("/api/v1/projects/{project_id}/documents/diff")
def compare_document_versions(
    project_id: str,
    from_document_id: str,
    to_document_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    """Compare two stored versions of one same-named document.

    This endpoint is deliberately local and deterministic: it never invokes the
    extraction provider and therefore cannot consume model tokens.
    """
    with SessionLocal() as db:
        if not _project_in_workspace(db, project_id, context.workspace_id):
            raise HTTPException(404, "项目不存在")
        old = db.scalar(
            select(DocumentRow)
            .join(ProjectRow, ProjectRow.id == DocumentRow.project_id)
            .where(
                DocumentRow.id == from_document_id,
                DocumentRow.project_id == project_id,
                ProjectRow.workspace_id == context.workspace_id,
            )
        )
        new = db.scalar(
            select(DocumentRow)
            .join(ProjectRow, ProjectRow.id == DocumentRow.project_id)
            .where(
                DocumentRow.id == to_document_id,
                DocumentRow.project_id == project_id,
                ProjectRow.workspace_id == context.workspace_id,
            )
        )
        if not old:
            raise HTTPException(404, "起始文档版本不存在于当前项目")
        if not new:
            raise HTTPException(404, "目标文档版本不存在于当前项目")
        if old.id == new.id:
            raise HTTPException(409, "请选择两个不同版本进行比较")
        if old.name.casefold() != new.name.casefold():
            raise HTTPException(409, "只能比较当前项目内同名文档的不同版本")

        diff = build_document_diff(
            old.content,
            new.content,
            max_lines=settings.diff_max_lines_per_version,
            max_chars=settings.diff_max_chars_per_version,
            max_output_lines=settings.diff_max_output_lines,
        )
        return {
            "from_document": {
                **serialize_document(old, include_content=False, db=db),
                "char_count": len(old.content),
                "line_count": len(old.content.splitlines()),
            },
            "to_document": {
                **serialize_document(new, include_content=False, db=db),
                "char_count": len(new.content),
                "line_count": len(new.content.splitlines()),
            },
            **diff,
        }


@app.get(
    "/api/v1/projects/{project_id}/documents/{document_id}/narrative-context"
)
def get_document_narrative_context(
    project_id: str,
    document_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    with SessionLocal() as db:
        document = db.scalar(
            select(DocumentRow)
            .join(ProjectRow, ProjectRow.id == DocumentRow.project_id)
            .where(
                DocumentRow.id == document_id,
                DocumentRow.project_id == project_id,
                ProjectRow.workspace_id == context.workspace_id,
            )
        )
        if document is None:
            raise HTTPException(404, "文档不存在")
        rows = list(
            db.scalars(
                select(DocumentNarrativeContextRevisionRow)
                .where(
                    DocumentNarrativeContextRevisionRow.project_id == project_id,
                    DocumentNarrativeContextRevisionRow.document_id == document_id,
                )
                .order_by(DocumentNarrativeContextRevisionRow.revision.desc())
            ).all()
        )
        legacy = db.get(DocumentContextRow, document_id)
        role = legacy.document_role if legacy else DEFAULT_DOCUMENT_ROLE
        story_scope = legacy.story_scope if legacy else DEFAULT_STORY_SCOPE
        if not rows:
            return {
                "document_id": document_id,
                "current": context_snapshot_payload(
                    None,
                    document_role=role,
                    story_scope=story_scope,
                ),
                "revision_count": 0,
                "revisions": [],
            }
        return {
            "document_id": document_id,
            "current": serialize_narrative_context_revision(rows[0]),
            "revision_count": len(rows),
            "revisions": [serialize_narrative_context_revision(row) for row in rows],
        }


def _narrative_context_inference_provider() -> OpenAICompatibleProvider:
    completion_caps = [
        settings.narrative_context_inference_max_completion_tokens
    ]
    if settings.provider_max_completion_tokens is not None:
        completion_caps.append(settings.provider_max_completion_tokens)
    response_caps = [settings.narrative_context_inference_max_response_bytes]
    if settings.provider_max_response_bytes is not None:
        response_caps.append(settings.provider_max_response_bytes)
    deadline_caps = [
        settings.narrative_context_inference_total_deadline_seconds
    ]
    if settings.provider_total_deadline_seconds is not None:
        deadline_caps.append(settings.provider_total_deadline_seconds)
    total_deadline = min(deadline_caps)
    bounded = settings.model_copy(
        update={
            # A local capability switch is required by the generic gateway;
            # it does not enable the extraction pipeline or mutate globals.
            "enable_model_extraction": True,
            "provider_timeout_seconds": min(
                settings.provider_timeout_seconds,
                settings.narrative_context_inference_timeout_seconds,
                total_deadline,
            ),
            "provider_total_deadline_seconds": total_deadline,
            "provider_max_attempts": 1,
            "provider_max_completion_tokens": min(completion_caps),
            "provider_max_response_bytes": min(response_caps),
        }
    )
    return OpenAICompatibleProvider(
        bounded,
        retry_policy=RetryPolicy(
            max_attempts=1,
            base_delay_seconds=0,
            jitter_ratio=0,
        ),
    )


def _narrative_context_inference_provider_error(exc: ProviderError) -> HTTPException:
    category = exc.category
    if category == "not_configured":
        return HTTPException(
            503,
            detail={
                "code": "context_inference_not_configured",
                "message": "模型尚未配置，请先在服务端配置可用的模型密钥",
            },
        )
    if category == "rate_limit":
        return HTTPException(
            429,
            detail={
                "code": "context_inference_rate_limited",
                "message": "模型服务当前限流，请稍后重试",
            },
        )
    if category in {"connect_timeout", "read_timeout", "transport"}:
        return HTTPException(
            504,
            detail={
                "code": "context_inference_timeout",
                "message": "模型服务暂时不可用或响应超时，请稍后重试",
            },
        )
    if category in {"unauthorized", "forbidden"}:
        return HTTPException(
            503,
            detail={
                "code": "context_inference_credentials_rejected",
                "message": "模型凭据不可用，请检查服务端模型配置",
            },
        )
    return HTTPException(
        502,
        detail={
            "code": "context_inference_provider_error",
            "message": "模型未能生成上下文建议，请稍后重试",
        },
    )


@app.post(
    "/api/v1/projects/{project_id}/documents/{document_id}"
    "/narrative-context/inference",
    status_code=201,
)
def infer_document_narrative_context(
    project_id: str,
    document_id: str,
    payload: NarrativeContextInferenceIn,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    """Suggest context from one immutable active body; never confirm authority."""

    # Phase 1 is read-only and closes before any network activity.
    with SessionLocal() as db:
        project = _project_in_workspace(db, project_id, context.workspace_id)
        if project is None:
            raise HTTPException(404, "项目不存在")
        document = db.scalar(
            select(DocumentRow).where(
                DocumentRow.id == document_id,
                DocumentRow.project_id == project_id,
                DocumentRow.active.is_(True),
            )
        )
        if document is None:
            raise HTTPException(404, "文档不存在")
        latest = latest_context_revisions(db, [document_id]).get(document_id)
        actual_revision = latest.revision if latest is not None else 0
        if payload.expected_revision != actual_revision:
            raise HTTPException(
                409,
                detail={
                    "code": "narrative_context_revision_conflict",
                    "message": "叙事上下文已被更新，请刷新后重试",
                    "actual_revision": actual_revision,
                },
            )
        if latest is not None and latest.resolution_state == "confirmed":
            raise HTTPException(
                409,
                detail={
                    "code": "context_inference_already_confirmed",
                    "message": "资料上下文已由用户确认；如需修改，请使用人工修订而非 AI 建议覆盖",
                    "actual_revision": actual_revision,
                },
            )
        frozen = {
            "version": document.version,
            "name": document.name,
            "content": document.content,
            "content_sha256": document_content_sha256(document.content),
        }

    if len(frozen["content"]) > MAX_INFERENCE_INPUT_CHARS:
        raise HTTPException(
            413,
            detail={
                "code": "context_inference_input_too_large",
                "message": "文档过长，暂不适合自动识别，请先手动设置资料上下文",
                "max_chars": MAX_INFERENCE_INPUT_CHARS,
            },
        )
    provider = _narrative_context_inference_provider()
    try:
        suggestion = infer_narrative_context(
            document_id=document_id,
            document_name=str(frozen["name"]),
            content=str(frozen["content"]),
            provider=provider,
        )
    except NarrativeContextInferenceInputError:
        raise HTTPException(
            422,
            detail={
                "code": "context_inference_input_invalid",
                "message": "文档内容为空或无法在安全上限内分析",
            },
        ) from None
    except NarrativeContextInferenceOutputError:
        raise HTTPException(
            502,
            detail={
                "code": "context_inference_invalid_output",
                "message": "模型返回的建议缺少有效结构或原文证据，请重试或手动设置",
            },
        ) from None
    except ProviderError as exc:
        raise _narrative_context_inference_provider_error(exc) from None

    evidence_payload = [
        {
            **span.model_dump(mode="json"),
            "supported_fields": support["supported_fields"],
        }
        for span, support in zip(
            suggestion.evidence, suggestion.evidence_support, strict=True
        )
    ]

    # Phase 3 obtains locks only after the provider call and revalidates every
    # frozen identity before one atomic role/context write.
    with SessionLocal() as db:
        project = db.scalar(
            select(ProjectRow)
            .where(
                ProjectRow.id == project_id,
                ProjectRow.workspace_id == context.workspace_id,
            )
            .with_for_update()
        )
        if project is None:
            raise HTTPException(404, "项目不存在")
        current = db.scalar(
            select(DocumentRow)
            .where(
                DocumentRow.id == document_id,
                DocumentRow.project_id == project_id,
                DocumentRow.active.is_(True),
            )
            .with_for_update()
        )
        if (
            current is None
            or current.version != frozen["version"]
            or current.name != frozen["name"]
            or document_content_sha256(current.content) != frozen["content_sha256"]
        ):
            raise HTTPException(
                409,
                detail={
                    "code": "context_inference_document_changed",
                    "message": "模型分析期间文档已更新，本次建议未保存，请重新识别",
                },
            )
        latest = latest_context_revisions(db, [document_id]).get(document_id)
        actual_revision = latest.revision if latest is not None else 0
        if payload.expected_revision != actual_revision:
            raise HTTPException(
                409,
                detail={
                    "code": "narrative_context_revision_conflict",
                    "message": "模型分析期间叙事上下文已更新，本次建议未保存",
                    "actual_revision": actual_revision,
                },
            )
        legacy = db.get(DocumentContextRow, document_id)
        if legacy is None:
            legacy = DocumentContextRow(
                document_id=document_id,
                document_role=suggestion.document_role,
                story_scope=DEFAULT_STORY_SCOPE,
            )
            db.add(legacy)
        else:
            legacy.document_role = suggestion.document_role
        try:
            row = add_context_revision(
                db,
                project_id=project_id,
                document_id=document_id,
                document_role=suggestion.document_role,
                resolution_state="inferred",
                publication_status=suggestion.publication_status,
                scope=suggestion.scope,
                origin="model_inferred",
                created_by_user_id=context.user_id,
                expected_revision=payload.expected_revision,
                inference_confidence=suggestion.confidence,
                inference_reasoning=suggestion.reasoning,
                inference_evidence=evidence_payload,
                inference_usage=suggestion.usage,
            )
            db.commit()
        except NarrativeContextRevisionConflict as exc:
            db.rollback()
            raise HTTPException(
                409,
                detail={
                    "code": "narrative_context_revision_conflict",
                    "message": "模型分析期间叙事上下文已更新，本次建议未保存",
                    "actual_revision": exc.actual_revision,
                },
            ) from None
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                409,
                detail={
                    "code": "narrative_context_revision_conflict",
                    "message": "模型分析期间叙事上下文已更新，本次建议未保存",
                },
            ) from None
        return {
            "document_id": document_id,
            "document_role": suggestion.document_role,
            "suggestion": serialize_narrative_context_revision(row),
            "usage": suggestion.usage,
        }


@app.post(
    "/api/v1/projects/{project_id}/documents/{document_id}/narrative-context/revisions",
    status_code=201,
)
def create_document_narrative_context_revision(
    project_id: str,
    document_id: str,
    payload: NarrativeContextRevisionInput,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    with SessionLocal() as db:
        project = db.scalar(
            select(ProjectRow)
            .where(
                ProjectRow.id == project_id,
                ProjectRow.workspace_id == context.workspace_id,
            )
            .with_for_update()
        )
        if project is None:
            raise HTTPException(404, "项目不存在")
        document = db.scalar(
            select(DocumentRow).where(
                DocumentRow.id == document_id,
                DocumentRow.project_id == project_id,
            )
        )
        if document is None:
            raise HTTPException(404, "文档不存在")
        legacy = db.get(DocumentContextRow, document_id)
        role = (
            payload.document_role
            or (legacy.document_role if legacy else DEFAULT_DOCUMENT_ROLE)
        )
        try:
            if legacy is None:
                legacy = DocumentContextRow(
                    document_id=document_id,
                    document_role=role,
                    story_scope=DEFAULT_STORY_SCOPE,
                )
                db.add(legacy)
            else:
                legacy.document_role = role
            row = add_context_revision(
                db,
                project_id=project_id,
                document_id=document_id,
                document_role=role,
                resolution_state=payload.resolution_state,
                publication_status=payload.publication_status,
                scope=payload.scope,
                origin="explicit",
                created_by_user_id=context.user_id,
                expected_revision=payload.expected_revision,
            )
            db.commit()
        except NarrativeContextRevisionConflict as exc:
            db.rollback()
            raise HTTPException(
                409,
                detail={
                    "code": "narrative_context_revision_conflict",
                    "message": "叙事上下文已被更新，请刷新后重试",
                    "actual_revision": exc.actual_revision,
                },
            ) from None
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                409,
                detail={
                    "code": "narrative_context_revision_conflict",
                    "message": "叙事上下文已被更新，请刷新后重试",
                },
            ) from None
        return {
            **serialize_narrative_context_revision(row),
            "document_role": role,
        }


@app.post("/api/v1/projects/{project_id}/documents/text", status_code=201)
def create_text_document(
    project_id: str,
    payload: TextDocumentIn,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    if not payload.name.lower().endswith((".md", ".txt", ".json")):
        raise HTTPException(415, "名称必须以 .md、.txt 或 .json 结尾")
    if len(payload.content.encode("utf-8")) > settings.max_upload_bytes:
        raise HTTPException(413, "文本超过上传限制")
    with SessionLocal() as db:
        if not _project_in_workspace(db, project_id, context.workspace_id):
            raise HTTPException(404, "项目不存在")
        version, superseded, document_role, story_scope = prepare_document_version(
            db,
            project_id,
            payload.name,
            payload.replace_document_id,
            payload.document_role,
            payload.story_scope,
        )
        row = DocumentRow(project_id=project_id, name=payload.name, content=payload.content, version=version)
        try:
            db.add(row)
            db.flush()
            db.add(
                DocumentContextRow(
                    document_id=row.id,
                    document_role=document_role,
                    story_scope=story_scope,
                )
            )
            add_initial_narrative_context(
                db,
                project_id=project_id,
                document_id=row.id,
                document_role=document_role,
                narrative_context=payload.narrative_context,
                user_id=context.user_id,
            )
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                409,
                detail={
                    "code": "document_version_conflict",
                    "message": "同名文档版本正在被更新，请刷新后重试",
                },
            ) from None
        return {**serialize_document(row, db=db), "superseded_document_ids": superseded}


@app.post("/api/v1/demo", status_code=201)
def create_demo(
    context: AuthContext = Depends(require_csrf),
) -> dict:
    data_dir = Path(__file__).resolve().parents[1] / "data" / "demo-natural"
    files = [
        (data_dir / "world.md", DocumentRole.canon),
        (data_dir / "chapter-01.md", DocumentRole.chapter),
    ]
    if not all(path.exists() for path, _ in files):
        raise HTTPException(500, "演示数据缺失")
    with SessionLocal() as db:
        project = ProjectRow(
            workspace_id=context.workspace_id,
            name="潮汐之门 · 自然文本体验",
            description="无需 API Key 的中文自然文本基线",
        )
        db.add(project); db.flush()
        for path, role in files:
            document = DocumentRow(project_id=project.id, name=path.name, content=path.read_text(encoding="utf-8"))
            db.add(document)
            db.flush()
            db.add(
                DocumentContextRow(
                    document_id=document.id,
                    document_role=role.value,
                    story_scope=DEFAULT_STORY_SCOPE,
                )
            )
            add_initial_narrative_context(
                db,
                project_id=project.id,
                document_id=document.id,
                document_role=role.value,
                narrative_context=None,
                user_id=context.user_id,
            )
        db.commit()
        return {"id": project.id, "name": project.name, "document_count": len(files)}


@app.post("/api/v1/demo/advanced", status_code=201)
def create_advanced_demo(
    context: AuthContext = Depends(require_csrf),
) -> dict:
    """Create the original multi-document acceptance scenario."""
    data_dir = Path(__file__).resolve().parents[1] / "data" / "advanced"
    files = [
        (data_dir / "world.md", DocumentRole.canon),
        (data_dir / "chapter-01.md", DocumentRole.chapter),
        (data_dir / "chapter-02.md", DocumentRole.chapter),
    ]
    if not all(path.exists() for path, _ in files):
        raise HTTPException(500, "复杂演示数据缺失")
    with SessionLocal() as db:
        project = ProjectRow(
            workspace_id=context.workspace_id,
            name="静默海域 · 复杂多章节验收",
            description="原创三文档场景，覆盖时间、地点、知识、物品与世界规则",
        )
        db.add(project); db.flush()
        for path, role in files:
            document = DocumentRow(project_id=project.id, name=path.name, content=path.read_text(encoding="utf-8"))
            db.add(document)
            db.flush()
            db.add(
                DocumentContextRow(
                    document_id=document.id,
                    document_role=role.value,
                    story_scope=DEFAULT_STORY_SCOPE,
                )
            )
            add_initial_narrative_context(
                db,
                project_id=project.id,
                document_id=document.id,
                document_role=role.value,
                narrative_context=None,
                user_id=context.user_id,
            )
        db.commit()
        return {"id": project.id, "name": project.name, "document_count": len(files)}


@app.post("/api/v1/projects/{project_id}/documents", status_code=201)
async def upload_document(
    project_id: str,
    file: UploadFile = File(...),
    replace_document_id: str | None = Form(None),
    document_role: DocumentRole | None = Form(None),
    story_scope: str | None = Form(
        None,
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9_\-\u4e00-\u9fff]+$",
    ),
    narrative_context: str | None = Form(None, max_length=8_000),
    context: AuthContext = Depends(require_csrf),
) -> dict:
    parsed_narrative_context = parse_multipart_narrative_context(
        narrative_context
    )
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "文件超过上传限制")
    if not file.filename or not file.filename.lower().endswith((".md", ".txt", ".json", ".docx")):
        raise HTTPException(415, "仅支持 Markdown、TXT、JSON 与标准 DOCX")
    if file.filename.lower().endswith(".docx"):
        try:
            content = extract_docx_text(data, max_text_bytes=settings.max_upload_bytes)
        except DocxImportError as exc:
            raise HTTPException(exc.status_code, str(exc)) from None
    else:
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "Markdown、TXT 与 JSON 文件必须为 UTF-8 编码") from None
    with SessionLocal() as db:
        if not _project_in_workspace(db, project_id, context.workspace_id):
            raise HTTPException(404, "项目不存在")
        version, superseded, resolved_role, resolved_scope = prepare_document_version(
            db,
            project_id,
            file.filename,
            replace_document_id,
            document_role,
            story_scope,
        )
        row = DocumentRow(project_id=project_id, name=file.filename, content=content, version=version)
        try:
            db.add(row)
            db.flush()
            db.add(
                DocumentContextRow(
                    document_id=row.id,
                    document_role=resolved_role,
                    story_scope=resolved_scope,
                )
            )
            add_initial_narrative_context(
                db,
                project_id=project_id,
                document_id=row.id,
                document_role=resolved_role,
                narrative_context=parsed_narrative_context,
                user_id=context.user_id,
            )
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                409,
                detail={
                    "code": "document_version_conflict",
                    "message": "同名文档版本正在被更新，请刷新后重试",
                },
            ) from None
        return {**serialize_document(row, db=db), "superseded_document_ids": superseded}


def _candidate_in_workspace(
    db,
    *,
    project_id: str,
    candidate_id: str,
    character_key: str,
    workspace_id: str,
    for_update: bool = False,
) -> CharacterTraitCandidateRow | None:
    statement = (
        select(CharacterTraitCandidateRow)
        .join(ProjectRow, ProjectRow.id == CharacterTraitCandidateRow.project_id)
        .where(
            CharacterTraitCandidateRow.id == candidate_id,
            CharacterTraitCandidateRow.project_id == project_id,
            CharacterTraitCandidateRow.character_key == character_key,
            ProjectRow.workspace_id == workspace_id,
        )
    )
    if for_update:
        statement = statement.with_for_update()
    return db.scalar(statement)


def _release_ranges_overlap(
    first: CharacterTraitCandidateRow, second: CharacterTraitCandidateRow
) -> bool:
    first_start = first.valid_from_release_ordinal or 0
    second_start = second.valid_from_release_ordinal or 0
    first_end = (
        first.valid_until_release_ordinal
        if first.valid_until_release_ordinal is not None
        else 2_147_483_647
    )
    second_end = (
        second.valid_until_release_ordinal
        if second.valid_until_release_ordinal is not None
        else 2_147_483_647
    )
    return max(first_start, second_start) <= min(first_end, second_end)


def _character_coverage_for_run(
    db, run: AnalysisRunRow | None
) -> tuple[str, str | None]:
    if run is None:
        return "unknown", None
    diagnostic = db.get(AnalysisDiagnosticRow, run.id)
    payload = diagnostic.payload if diagnostic and isinstance(diagnostic.payload, dict) else {}
    stage = payload.get("character_consistency")
    if not isinstance(stage, dict):
        return "unknown", "该次分析没有可核对的角色一致性执行记录"
    outcome = stage.get("outcome")
    if outcome == "completed":
        return "full", "角色一致性阶段已完整执行"
    if outcome == "partial":
        return "partial", "仅完成部分材料审查；未覆盖内容不能视为没有问题"
    if outcome == "degraded":
        return "unknown", "模型阶段已降级；当前结果不能代表角色审查完成"
    if outcome == "skipped":
        return "unknown", "角色一致性阶段未执行；当前结果不能代表没有问题"
    return "unknown", "角色一致性执行状态无法确认"


@app.get("/api/v1/projects/{project_id}/characters")
def list_characters(
    project_id: str,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 40,
    query: Annotated[str | None, Query(max_length=160)] = None,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    with SessionLocal() as db:
        if not _project_in_workspace(db, project_id, context.workspace_id):
            raise HTTPException(404, "项目不存在")
        active_document_count = db.scalar(
            select(func.count())
            .select_from(DocumentRow)
            .where(
                DocumentRow.project_id == project_id,
                DocumentRow.active.is_(True),
            )
        ) or 0
        latest_completed_run = db.scalar(
            select(AnalysisRunRow)
            .where(
                AnalysisRunRow.project_id == project_id,
                AnalysisRunRow.status == "completed",
                AnalysisRunRow.batch_mode == "baseline_build",
            )
            .order_by(
                AnalysisRunRow.completed_at.desc(),
                AnalysisRunRow.created_at.desc(),
                AnalysisRunRow.id.desc(),
            )
            .limit(1)
        )
        if latest_completed_run is None:
            # Compatibility for projects that created their character baseline
            # before Guided Review Batch existed.  A completed draft_review is
            # never a baseline, and the fallback is permanently disabled as
            # soon as the project has any completed baseline_build.
            latest_completed_run = db.scalar(
                select(AnalysisRunRow)
                .where(
                    AnalysisRunRow.project_id == project_id,
                    AnalysisRunRow.status == "completed",
                    or_(
                        AnalysisRunRow.batch_mode == "full_review",
                        AnalysisRunRow.batch_mode.is_(None),
                    ),
                )
                .order_by(
                    AnalysisRunRow.completed_at.desc(),
                    AnalysisRunRow.created_at.desc(),
                    AnalysisRunRow.id.desc(),
                )
                .limit(1)
            )
        rows = (
            list(
                db.scalars(
                    select(CharacterTraitCandidateRow)
                    .where(
                        CharacterTraitCandidateRow.project_id == project_id,
                        CharacterTraitCandidateRow.source_run_id
                        == latest_completed_run.id,
                    )
                    .order_by(
                        CharacterTraitCandidateRow.character_key,
                        CharacterTraitCandidateRow.created_at,
                    )
                ).all()
            )
            if latest_completed_run is not None
            else []
        )
        grouped: dict[str, dict] = {}
        for row in rows:
            item = grouped.setdefault(
                row.character_key,
                {
                    "character_key": row.character_key,
                    "character_display_name": row.character_display_name,
                    "confirmed_trait_count": 0,
                    "pending_candidate_count": 0,
                    "profile_revision": 0,
                    "updated_at": row.created_at,
                },
            )
            if row.review_state == "confirmed":
                item["confirmed_trait_count"] += 1
            elif row.review_state == "pending":
                item["pending_candidate_count"] += 1
            item["profile_revision"] = max(
                item["profile_revision"], row.lock_version
            )
            candidate_updated_at = row.reviewed_at or row.created_at
            if candidate_updated_at > item["updated_at"]:
                item["updated_at"] = candidate_updated_at
        items = list(grouped.values())
        normalized_query = (query or "").strip().casefold()
        if normalized_query:
            items = [
                item
                for item in items
                if normalized_query
                in item["character_display_name"].casefold()
            ]
        total = len(items)
        offset = (page - 1) * page_size
        items = items[offset : offset + page_size]
        if not active_document_count:
            readiness = "no_documents"
        elif latest_completed_run is None:
            readiness = "no_completed_run"
        elif not grouped:
            readiness = "not_generated"
        else:
            readiness = "ready"
        model_coverage, coverage_detail = _character_coverage_for_run(
            db, latest_completed_run
        )
        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_more": offset + len(items) < total,
            "readiness": readiness,
            "model_coverage": model_coverage,
            "coverage_detail": coverage_detail,
            "source_run_id": (
                latest_completed_run.id if latest_completed_run else None
            ),
        }


@app.get("/api/v1/projects/{project_id}/characters/{character_key}")
def get_character_profile(
    project_id: str,
    character_key: str,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    try:
        normalized = normalize_character_key(character_key)
    except ValueError:
        raise HTTPException(404, "角色不存在") from None
    with SessionLocal() as db:
        if not _project_in_workspace(db, project_id, context.workspace_id):
            raise HTTPException(404, "项目不存在")
        rows = list(
            db.scalars(
                select(CharacterTraitCandidateRow)
                .where(
                    CharacterTraitCandidateRow.project_id == project_id,
                    CharacterTraitCandidateRow.character_key == normalized,
                )
                .order_by(
                    CharacterTraitCandidateRow.trait_type,
                    CharacterTraitCandidateRow.trait_key,
                    CharacterTraitCandidateRow.created_at,
                )
            ).all()
        )
        if not rows:
            raise HTTPException(404, "角色不存在")
        return {
            "character_key": normalized,
            "character_display_name": rows[-1].character_display_name,
            "confirmed_traits": [
                serialize_character_trait_candidate(row)
                for row in rows
                if row.review_state == "confirmed"
            ],
            "pending_candidate_count": sum(
                row.review_state == "pending" for row in rows
            ),
        }


@app.get(
    "/api/v1/projects/{project_id}/characters/{character_key}/profile-candidates"
)
def list_character_profile_candidates(
    project_id: str,
    character_key: str,
    state: Literal["pending", "confirmed", "rejected", "superseded"] | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    try:
        normalized = normalize_character_key(character_key)
    except ValueError:
        raise HTTPException(404, "角色不存在") from None
    with SessionLocal() as db:
        if not _project_in_workspace(db, project_id, context.workspace_id):
            raise HTTPException(404, "项目不存在")
        filters = [
            CharacterTraitCandidateRow.project_id == project_id,
            CharacterTraitCandidateRow.character_key == normalized,
        ]
        if state is not None:
            filters.append(CharacterTraitCandidateRow.review_state == state)
        total = db.scalar(
            select(func.count())
            .select_from(CharacterTraitCandidateRow)
            .where(*filters)
        ) or 0
        rows = list(
            db.scalars(
                select(CharacterTraitCandidateRow)
                .where(*filters)
                .order_by(
                    CharacterTraitCandidateRow.created_at,
                    CharacterTraitCandidateRow.id,
                )
                .offset(offset)
                .limit(limit)
            ).all()
        )
        if total == 0:
            # Do not reveal whether a differently-normalized/cross-project
            # candidate exists; a project with no such character is a 404.
            any_character = db.scalar(
                select(CharacterTraitCandidateRow.id)
                .where(
                    CharacterTraitCandidateRow.project_id == project_id,
                    CharacterTraitCandidateRow.character_key == normalized,
                )
                .limit(1)
            )
            if any_character is None:
                raise HTTPException(404, "角色不存在")
        return {
            "character_key": normalized,
            "state": state,
            "limit": limit,
            "offset": offset,
            "total": total,
            "has_more": offset + len(rows) < total,
            "items": [
                serialize_character_trait_candidate_for_review(db, row)
                for row in rows
            ],
            "model_coverage": "unknown",
            "coverage_detail": (
                "候选的模型覆盖度以其来源运行诊断为准；确认前请核对证据"
            ),
        }


@app.get(
    "/api/v1/projects/{project_id}/characters/{character_key}/profile-candidates/{candidate_id}"
)
def get_character_profile_candidate(
    project_id: str,
    character_key: str,
    candidate_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    try:
        normalized = normalize_character_key(character_key)
    except ValueError:
        raise HTTPException(404, "角色候选不存在") from None
    with SessionLocal() as db:
        row = _candidate_in_workspace(
            db,
            project_id=project_id,
            candidate_id=candidate_id,
            character_key=normalized,
            workspace_id=context.workspace_id,
        )
        if row is None:
            raise HTTPException(404, "角色候选不存在")
        reviews = list(
            db.scalars(
                select(CharacterTraitReviewRow)
                .where(CharacterTraitReviewRow.candidate_id == row.id)
                .order_by(
                    CharacterTraitReviewRow.created_at,
                    CharacterTraitReviewRow.id,
                )
            ).all()
        )
        return {
            **serialize_character_trait_candidate_for_review(db, row),
            "decisions": [
                {
                    "id": review.id,
                    "decision": review.decision,
                    "expected_revision": review.expected_lock_version,
                    "comment": review.comment,
                    "created_at": review.created_at,
                }
                for review in reviews
            ],
        }


@app.post(
    "/api/v1/projects/{project_id}/characters/{character_key}/profile-candidates/{candidate_id}/decisions",
    status_code=201,
)
def decide_character_profile_candidate(
    project_id: str,
    character_key: str,
    candidate_id: str,
    payload: CharacterTraitDecisionIn,
    idempotency_key_header: Annotated[
        str | None, Header(alias="Idempotency-Key", max_length=128)
    ] = None,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    idempotency_key = _normalize_idempotency_key(idempotency_key_header)
    try:
        normalized = normalize_character_key(character_key)
    except ValueError:
        raise HTTPException(404, "角色候选不存在") from None
    with SessionLocal() as db:
        project = db.scalar(
            select(ProjectRow)
            .where(
                ProjectRow.id == project_id,
                ProjectRow.workspace_id == context.workspace_id,
            )
            .with_for_update()
        )
        if project is None:
            raise HTTPException(404, "项目不存在")
        row = _candidate_in_workspace(
            db,
            project_id=project_id,
            candidate_id=candidate_id,
            character_key=normalized,
            workspace_id=context.workspace_id,
            for_update=True,
        )
        if row is None:
            raise HTTPException(404, "角色候选不存在")
        existing_review = (
            db.scalar(
                select(CharacterTraitReviewRow).where(
                    CharacterTraitReviewRow.candidate_id == candidate_id,
                    CharacterTraitReviewRow.idempotency_key == idempotency_key,
                )
            )
            if idempotency_key is not None
            else None
        )
        if existing_review is not None:
            if (
                existing_review.decision != payload.decision
                or existing_review.expected_lock_version
                != payload.expected_revision
                or existing_review.comment != payload.comment
            ):
                raise HTTPException(
                    409,
                    detail={
                        "code": "idempotency_key_conflict",
                        "message": "同一幂等键不能用于不同的候选审核请求",
                    },
                )
            return {
                "candidate": serialize_character_trait_candidate(row),
                "decision_id": existing_review.id,
                "deduplicated": True,
            }
        source_is_current, stale_reason = _candidate_source_is_current(db, row)
        if not source_is_current:
            raise HTTPException(
                409,
                detail={
                    "code": "character_trait_candidate_stale",
                    "message": stale_reason,
                },
            )
        if row.review_state != "pending" or row.lock_version != payload.expected_revision:
            raise HTTPException(
                409,
                detail={
                    "code": "character_trait_revision_conflict",
                    "message": "角色特征候选已被审核，请刷新后重试",
                    "actual_revision": row.lock_version,
                    "review_state": row.review_state,
                },
            )
        if payload.decision == "confirm":
            confirmed = list(
                db.scalars(
                    select(CharacterTraitCandidateRow).where(
                        CharacterTraitCandidateRow.project_id == project_id,
                        CharacterTraitCandidateRow.character_key == row.character_key,
                        CharacterTraitCandidateRow.trait_type == row.trait_type,
                        CharacterTraitCandidateRow.review_state == "confirmed",
                        CharacterTraitCandidateRow.id != row.id,
                    )
                ).all()
            )
            conflicts = [
                other
                for other in confirmed
                if trait_keys_compatible(
                    dimension=row.trait_type,
                    baseline_key=other.trait_key,
                    observation_key=row.trait_key,
                )
                if other.id != row.supersedes_candidate_id
                and other.value != row.value
                and _release_ranges_overlap(row, other)
                and scope_relation(
                    row.scope_payload,
                    other.scope_payload,
                    first_resolution="confirmed",
                    second_resolution="confirmed",
                )
                != "incompatible"
            ]
            if conflicts:
                raise HTTPException(
                    409,
                    detail={
                        "code": "character_trait_confirmation_conflict",
                        "message": "同一作用域内已有不同的已确认角色特征",
                    },
                )
            if row.supersedes_candidate_id:
                replaced = db.scalar(
                    select(CharacterTraitCandidateRow)
                    .where(
                        CharacterTraitCandidateRow.id
                        == row.supersedes_candidate_id,
                        CharacterTraitCandidateRow.project_id == project_id,
                    )
                    .with_for_update()
                )
                try:
                    if replaced is None:
                        raise ValueError("superseded candidate is missing")
                    validate_character_trait_supersession(
                        project_id=project_id,
                        character_key=row.character_key,
                        trait_type=row.trait_type,
                        trait_key=row.trait_key,
                        scope=row.scope_payload,
                        valid_from_release_ordinal=(
                            row.valid_from_release_ordinal
                        ),
                        valid_until_release_ordinal=(
                            row.valid_until_release_ordinal
                        ),
                        superseded=replaced,
                    )
                except ValueError:
                    raise HTTPException(
                        409,
                        detail={
                            "code": "character_trait_supersession_conflict",
                            "message": "待替代的角色特征已发生变化",
                        },
                    ) from None
                db.add(
                    CharacterTraitReviewRow(
                        project_id=project_id,
                        candidate_id=replaced.id,
                        decision="supersede",
                        expected_lock_version=replaced.lock_version,
                        idempotency_key=None,
                        comment=f"由候选 {row.id} 替代",
                        created_by_user_id=context.user_id,
                    )
                )
                replaced.review_state = "superseded"
                replaced.lock_version += 1
                replaced.reviewed_at = utc_now_naive()
                replaced.reviewed_by_user_id = context.user_id
        changed = db.execute(
            update(CharacterTraitCandidateRow)
            .where(
                CharacterTraitCandidateRow.id == row.id,
                CharacterTraitCandidateRow.project_id == project_id,
                CharacterTraitCandidateRow.review_state == "pending",
                CharacterTraitCandidateRow.lock_version == payload.expected_revision,
            )
            .values(
                review_state=(
                    "confirmed" if payload.decision == "confirm" else "rejected"
                ),
                lock_version=payload.expected_revision + 1,
                reviewed_at=utc_now_naive(),
                reviewed_by_user_id=context.user_id,
            )
        ).rowcount
        if changed != 1:
            db.rollback()
            raise HTTPException(
                409,
                detail={
                    "code": "character_trait_revision_conflict",
                    "message": "角色特征候选已被审核，请刷新后重试",
                },
            )
        review = CharacterTraitReviewRow(
            project_id=project_id,
            candidate_id=row.id,
            decision=payload.decision,
            expected_lock_version=payload.expected_revision,
            idempotency_key=idempotency_key,
            comment=payload.comment,
            created_by_user_id=context.user_id,
        )
        db.add(review)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                409,
                detail={
                    "code": "character_trait_revision_conflict",
                    "message": "角色特征候选已被审核，请刷新后重试",
                },
            ) from None
        db.refresh(row)
        return {
            "candidate": serialize_character_trait_candidate(row),
            "decision_id": review.id,
            "deduplicated": False,
        }


@app.get("/api/v1/projects/{project_id}/drift-issues")
def list_character_drift_issues(
    project_id: str,
    character_key: Annotated[
        str | None, Query(min_length=1, max_length=160)
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    normalized_character_key: str | None = None
    if character_key is not None:
        try:
            normalized_character_key = normalize_character_key(character_key)
        except ValueError:
            raise HTTPException(422, "角色标识无效") from None
    with SessionLocal() as db:
        if not _project_in_workspace(db, project_id, context.workspace_id):
            raise HTTPException(404, "项目不存在")
        filters = [
            AnalysisRunRow.project_id == project_id,
            IssueRow.category == "character_drift",
        ]
        if normalized_character_key is not None:
            filters.append(
                IssueRow.extra["character_key"].as_string()
                == normalized_character_key
            )
        total = db.scalar(
            select(func.count())
            .select_from(IssueRow)
            .join(AnalysisRunRow, AnalysisRunRow.id == IssueRow.run_id)
            .where(*filters)
        ) or 0
        rows = list(
            db.execute(
                select(IssueRow, AnalysisRunRow.id)
                .join(AnalysisRunRow, AnalysisRunRow.id == IssueRow.run_id)
                .where(*filters)
                .order_by(AnalysisRunRow.created_at.desc(), IssueRow.id)
                .offset(offset)
                .limit(limit)
            ).all()
        )
        latest_feedback: dict[str, str] = {}
        issue_ids = {issue.id for issue, _ in rows}
        if issue_ids:
            feedback_rows = list(
                db.scalars(
                    select(FeedbackRow)
                    .where(FeedbackRow.issue_id.in_(issue_ids))
                    .order_by(
                        FeedbackRow.created_at.desc(),
                        FeedbackRow.id.desc(),
                    )
                ).all()
            )
            for feedback_row in feedback_rows:
                latest_feedback.setdefault(
                    feedback_row.issue_id, feedback_row.label
                )
        return {
            "project_id": project_id,
            "limit": limit,
            "offset": offset,
            "total": total,
            "has_more": offset + len(rows) < total,
            "items": [
                {
                    "run_id": run_id,
                    **_serialize_issue(issue),
                    "feedback_status": latest_feedback.get(
                        issue.id, "unreviewed"
                    ),
                }
                for issue, run_id in rows
            ],
        }


@app.post("/api/v1/projects/{project_id}/analysis-runs", status_code=202)
def start_analysis(
    project_id: str,
    payload: AnalysisRunIn | None = None,
    idempotency_key_header: Annotated[
        str | None, Header(alias="Idempotency-Key", max_length=128)
    ] = None,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    request = payload or AnalysisRunIn()
    intent = _review_batch_intent(request)
    idempotency_key = _normalize_idempotency_key(idempotency_key_header)
    with SessionLocal() as db:
        project = db.scalar(
            select(ProjectRow)
            .where(
                ProjectRow.id == project_id,
                ProjectRow.workspace_id == context.workspace_id,
            )
            .with_for_update()
        )
        if project is None:
            raise HTTPException(404, "项目不存在")
        existing = _idempotent_run(db, project_id, idempotency_key)
        if existing is not None:
            _require_idempotency_operation(
                db,
                existing,
                retried_from_run_id=None,
                review_batch_intent=intent,
            )
            return _accepted_run_payload(db, existing, created=False)
        documents, batch_roles, coverage = _review_batch_selection(
            db,
            project_id=project_id,
            payload=request,
        )
        enforce_daily_model_budget(db, context.workspace_id)
        run = AnalysisRunRow(
            project_id=project_id,
            requested_by_user_id=context.user_id,
            idempotency_key=idempotency_key,
            batch_mode=request.mode,
            sensitivity=request.sensitivity,
            batch_coverage=coverage,
        )
        try:
            run, created = _create_run_or_load_winner(
                db,
                run,
                idempotency_key,
                lambda created_run: capture_run_inputs(
                    db,
                    created_run,
                    list(documents),
                    batch_roles=batch_roles,
                ),
            )
        except CharacterTraitSnapshotLimitExceeded:
            db.rollback()
            raise HTTPException(
                409,
                detail={
                    "code": "character_profile_snapshot_limit_exceeded",
                    "message": "已确认角色档案数量超过单次分析上限",
                },
            ) from None
        _require_idempotency_operation(
            db,
            run,
            retried_from_run_id=None,
            review_batch_intent=intent,
        )
        payload = _accepted_run_payload(db, run, created=created)
        run_id = run.id
    if created:
        _dispatch_created_run(run_id)
    return payload


@app.get("/api/v1/analysis-runs/{run_id}")
def get_run(
    run_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    with SessionLocal() as db:
        row = _run_in_workspace(db, run_id, context.workspace_id)
        if not row: raise HTTPException(404, "分析任务不存在")
        return serialize_run(row, db)


@app.get("/api/v1/projects/{project_id}/analysis-runs")
def list_analysis_runs(
    project_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> list[dict]:
    with SessionLocal() as db:
        if not _project_in_workspace(db, project_id, context.workspace_id):
            raise HTTPException(404, "项目不存在")
        rows = db.scalars(
            select(AnalysisRunRow)
            .where(AnalysisRunRow.project_id == project_id)
            .order_by(
                AnalysisRunRow.created_at.desc(),
                AnalysisRunRow.id.desc(),
            )
        ).all()
        return [serialize_run(row, db) for row in rows]


@app.post("/api/v1/analysis-runs/{run_id}/cancel", status_code=202)
def cancel_run(
    run_id: str,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    with SessionLocal() as db:
        row = _run_in_workspace(db, run_id, context.workspace_id)
        if not row: raise HTTPException(404, "分析任务不存在")
        if row.status == "cancelled" and row.cancel_requested:
            return {"id": row.id, "cancel_requested": True, "already_requested": True}
        if row.status in {"completed", "failed", "cancelled"}:
            raise HTTPException(409, f"终态任务不能取消：{row.status}")
        cancelled_before_start = db.execute(
            update(AnalysisRunRow)
            .where(
                AnalysisRunRow.id == run_id,
                AnalysisRunRow.status == "queued",
                AnalysisRunRow.cancel_requested.is_(False),
            )
            .values(
                status="cancelled",
                cancel_requested=True,
                completed_at=utc_now_naive(),
            )
        ).rowcount
        if cancelled_before_start == 1:
            db.add(
                RunEventRow(
                    run_id=run_id,
                    stage="cancelled",
                    progress=100,
                    message="任务在工作进程开始前已取消",
                )
            )
            db.commit()
            return {"id": row.id, "cancel_requested": True, "already_requested": False}
        changed = db.execute(
            update(AnalysisRunRow)
            .where(
                AnalysisRunRow.id == run_id,
                AnalysisRunRow.status.in_(("queued", "running")),
                AnalysisRunRow.cancel_requested.is_(False),
            )
            .values(cancel_requested=True)
        ).rowcount
        if changed != 1:
            db.rollback()
            db.refresh(row)
            if row.status == "cancelled" and row.cancel_requested:
                return {"id": row.id, "cancel_requested": True, "already_requested": True}
            if row.status in {"completed", "failed", "cancelled"}:
                raise HTTPException(409, f"终态任务不能取消：{row.status}")
            return {"id": row.id, "cancel_requested": True, "already_requested": True}
        db.commit()
        return {"id": row.id, "cancel_requested": True, "already_requested": False}


@app.post("/api/v1/analysis-runs/{run_id}/retry", status_code=202)
def retry_run(
    run_id: str,
    idempotency_key_header: Annotated[
        str | None, Header(alias="Idempotency-Key", max_length=128)
    ] = None,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    idempotency_key = _normalize_idempotency_key(idempotency_key_header)
    with SessionLocal() as db:
        old = _run_in_workspace(db, run_id, context.workspace_id)
        if not old: raise HTTPException(404, "分析任务不存在")
        existing = _idempotent_run(db, old.project_id, idempotency_key)
        if existing is not None:
            _require_idempotency_operation(
                db, existing, retried_from_run_id=run_id
            )
            return _accepted_run_payload(db, existing, created=False)
        if old.status not in {"failed", "cancelled"}:
            raise HTTPException(409, "仅失败或已取消任务可以重试")
        snapshot_count = db.scalar(
            select(func.count())
            .select_from(AnalysisRunInputRow)
            .where(AnalysisRunInputRow.run_id == run_id)
        )
        if not snapshot_count:
            raise HTTPException(409, MISSING_SNAPSHOT_ERROR)
        enforce_daily_model_budget(db, context.workspace_id)
        row = AnalysisRunRow(
            project_id=old.project_id,
            requested_by_user_id=context.user_id,
            idempotency_key=idempotency_key,
            batch_mode=old.batch_mode or "full_review",
            sensitivity=old.sensitivity or "balanced",
            batch_coverage=json.loads(
                json.dumps(old.batch_coverage or {}, ensure_ascii=False)
            ),
        )
        row, created = _create_run_or_load_winner(
            db,
            row,
            idempotency_key,
            lambda created_run: copy_run_inputs(db, old.id, created_run),
        )
        _require_idempotency_operation(
            db, row, retried_from_run_id=run_id
        )
        payload = _accepted_run_payload(db, row, created=created)
        new_id = row.id
    if created:
        _dispatch_created_run(new_id)
    return payload


@app.post("/api/v1/analysis-runs/{baseline_run_id}/rechecks", status_code=202)
def start_recheck(
    baseline_run_id: str,
    idempotency_key_header: Annotated[
        str | None, Header(alias="Idempotency-Key", max_length=128)
    ] = None,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    """Analyze the project's current revision against one frozen baseline."""
    idempotency_key = _normalize_idempotency_key(idempotency_key_header)
    with SessionLocal() as db:
        baseline = _run_in_workspace(db, baseline_run_id, context.workspace_id)
        if baseline is None:
            raise HTTPException(404, "基准分析任务不存在")
        existing = _idempotent_run(db, baseline.project_id, idempotency_key)
        if existing is not None:
            _require_idempotency_operation(
                db,
                existing,
                retried_from_run_id=None,
                recheck_baseline_run_id=baseline_run_id,
            )
            comparison = db.scalar(
                select(AnalysisRunComparisonRow).where(
                    AnalysisRunComparisonRow.target_run_id == existing.id
                )
            )
            payload = _accepted_run_payload(db, existing, created=False)
            return {
                **payload,
                "baseline_run_id": baseline_run_id,
                "comparison_id": comparison.id,
                "comparison_status": comparison.status,
            }
        if baseline.status != "completed":
            raise HTTPException(409, "只有已完成的分析任务可以作为复检基准")
        locked_project = db.scalar(
            select(ProjectRow)
            .where(
                ProjectRow.id == baseline.project_id,
                ProjectRow.workspace_id == context.workspace_id,
            )
            .with_for_update()
        )
        if locked_project is None:
            raise HTTPException(404, "项目不存在")
        baseline_inputs = list(
            db.scalars(
                select(AnalysisRunInputRow)
                .where(AnalysisRunInputRow.run_id == baseline_run_id)
                .order_by(AnalysisRunInputRow.ordinal)
            ).all()
        )
        if not baseline_inputs:
            raise HTTPException(409, MISSING_SNAPSHOT_ERROR)
        batch_mode = baseline.batch_mode or "full_review"
        sensitivity = baseline.sensitivity or "balanced"
        target_document_ids: list[str] | None = None
        if batch_mode == "draft_review":
            baseline_metadata = run_input_metadata(db, baseline_run_id)
            logical_target_names = {
                str(item["document_name"]).casefold()
                for item in baseline_metadata
                if item.get("batch_role") == "target"
            }
            if not logical_target_names:
                raise HTTPException(
                    409,
                    detail={
                        "code": "missing_draft_review_targets",
                        "message": "基准运行缺少可复检的冻结目标",
                    },
                )
            current_targets = list(
                db.scalars(
                    select(DocumentRow)
                    .where(
                        DocumentRow.project_id == baseline.project_id,
                        DocumentRow.active.is_(True),
                        func.lower(DocumentRow.name).in_(logical_target_names),
                    )
                    .order_by(DocumentRow.created_at, DocumentRow.id)
                ).all()
            )
            if {row.name.casefold() for row in current_targets} != logical_target_names:
                raise HTTPException(
                    409,
                    detail={
                        "code": "logical_target_missing",
                        "message": "草稿目标的当前活动版本缺失，无法复检",
                    },
                )
            target_document_ids = [row.id for row in current_targets]
        recheck_request = AnalysisRunIn(
            mode=batch_mode,
            sensitivity=sensitivity,
            target_document_ids=target_document_ids,
        )
        documents, batch_roles, coverage = _review_batch_selection(
            db,
            project_id=baseline.project_id,
            payload=recheck_request,
        )
        coverage["recheck_baseline_run_id"] = baseline_run_id
        try:
            baseline_signature = frozen_run_source_signature(
                db, baseline_run_id
            )
            current_signature = current_project_source_signature(
                db,
                baseline.project_id,
                documents,
                batch_roles=batch_roles,
            )
        except CharacterTraitSnapshotLimitExceeded:
            raise HTTPException(
                409,
                detail={
                    "code": "character_profile_snapshot_limit_exceeded",
                    "message": "已确认角色档案数量超过单次分析上限",
                },
            ) from None
        if baseline_signature == current_signature:
            raise HTTPException(
                409,
                detail={
                    "code": "no_document_changes",
                    "message": "当前活动文档与基准运行的冻结输入相同，请先保存新版本",
                },
            )
        enforce_daily_model_budget(db, context.workspace_id)
        row = AnalysisRunRow(
            project_id=baseline.project_id,
            requested_by_user_id=context.user_id,
            idempotency_key=idempotency_key,
            batch_mode=batch_mode,
            sensitivity=sensitivity,
            batch_coverage=coverage,
        )

        def prepare_recheck(created_run: AnalysisRunRow) -> None:
            capture_run_inputs(
                db,
                created_run,
                documents,
                batch_roles=batch_roles,
            )
            db.flush()
            target_inputs = list(
                db.scalars(
                    select(AnalysisRunInputRow)
                    .where(AnalysisRunInputRow.run_id == created_run.id)
                    .order_by(AnalysisRunInputRow.ordinal)
                ).all()
            )
            db.add(
                AnalysisRunComparisonRow(
                    project_id=baseline.project_id,
                    baseline_run_id=baseline_run_id,
                    target_run_id=created_run.id,
                    matcher_version=MATCHER_VERSION,
                    status="pending",
                    summary={},
                    provenance={
                        "matcher_version": MATCHER_VERSION,
                        "input_diff": build_input_diff(
                            baseline_inputs, target_inputs
                        ),
                        "semantic_equivalence_guaranteed": False,
                    },
                )
            )

        try:
            row, created = _create_run_or_load_winner(
                db, row, idempotency_key, prepare_recheck
            )
        except CharacterTraitSnapshotLimitExceeded:
            db.rollback()
            raise HTTPException(
                409,
                detail={
                    "code": "character_profile_snapshot_limit_exceeded",
                    "message": "已确认角色档案数量超过单次分析上限",
                },
            ) from None
        _require_idempotency_operation(
            db,
            row,
            retried_from_run_id=None,
            recheck_baseline_run_id=baseline_run_id,
        )
        comparison = db.scalar(
            select(AnalysisRunComparisonRow).where(
                AnalysisRunComparisonRow.target_run_id == row.id
            )
        )
        payload = {
            **_accepted_run_payload(db, row, created=created),
            "baseline_run_id": baseline_run_id,
            "comparison_id": comparison.id,
            "comparison_status": comparison.status,
        }
        target_run_id = row.id
    if created:
        _dispatch_created_run(target_run_id)
    return payload


def _serialize_issue(row: IssueRow | None) -> dict | None:
    if row is None:
        return None
    return {
        "id": row.id,
        "category": row.category,
        "severity": row.severity,
        "confidence": row.confidence,
        "title": row.title,
        "explanation": row.explanation,
        "evidence": row.evidence,
        "suggestion": row.suggestion,
        "metadata": row.extra,
    }


def _comparison_feedback_snapshot(provenance: object) -> dict | None:
    if not isinstance(provenance, dict):
        return None
    snapshot = provenance.get("baseline_feedback_snapshot")
    if not isinstance(snapshot, dict):
        return None
    identifier = snapshot.get("id")
    label = snapshot.get("label")
    comment = snapshot.get("comment")
    created_at = snapshot.get("created_at")
    if (
        not isinstance(identifier, str)
        or label not in {"accepted", "false_positive", "resolved"}
        or not isinstance(comment, str)
        or not isinstance(created_at, str)
    ):
        return None
    return {
        "id": identifier,
        "label": label,
        "comment": comment,
        "created_at": created_at,
    }


@app.get("/api/v1/analysis-runs/{target_run_id}/comparison")
def get_run_comparison(
    target_run_id: str,
    outcome: ComparisonOutcome | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    with SessionLocal() as db:
        target = _run_in_workspace(db, target_run_id, context.workspace_id)
        if target is None:
            raise HTTPException(404, "分析任务不存在")
        comparison = db.scalar(
            select(AnalysisRunComparisonRow).where(
                AnalysisRunComparisonRow.target_run_id == target_run_id
            )
        )
        if comparison is None:
            raise HTTPException(404, "该分析任务不是复检任务")
        baseline = _run_in_workspace(
            db, comparison.baseline_run_id, context.workspace_id
        )
        if (
            baseline is None
            or baseline.project_id != target.project_id
            or comparison.project_id != target.project_id
        ):
            # Fail closed if persisted lineage was corrupted or manually
            # inserted across projects; never follow it into another workspace.
            raise HTTPException(404, "复检谱系不存在")
        if target.status == "completed" and comparison.status != "ready":
            try:
                with db.begin_nested():
                    comparison = materialize_run_comparison(db, target_run_id)
            except Exception:
                with db.begin_nested():
                    comparison = mark_comparison_unverifiable(db, target_run_id)
            db.commit()

        effective_status = (
            "ready"
            if comparison.status == "ready"
            else target.status
            if target.status in {"failed", "cancelled"}
            else "pending"
        )
        payload = {
            "id": comparison.id,
            "project_id": comparison.project_id,
            "baseline_run_id": comparison.baseline_run_id,
            "target_run_id": comparison.target_run_id,
            "status": effective_status,
            "target_run_status": target.status,
            "matcher": {
                "version": comparison.matcher_version,
                "kind": "deterministic_heuristic",
                "semantic_equivalence_guaranteed": False,
            },
            "summary": comparison.summary if comparison.status == "ready" else None,
            "provenance": comparison.provenance,
            "created_at": comparison.created_at,
            "completed_at": comparison.completed_at,
            "page": {
                "outcome": outcome,
                "limit": limit,
                "offset": offset,
                "returned": 0,
                "total": 0,
                "has_more": False,
            },
            "items": [],
        }
        if comparison.status != "ready":
            return payload
        item_filter = [IssueComparisonItemRow.comparison_id == comparison.id]
        if outcome is not None:
            item_filter.append(IssueComparisonItemRow.outcome == outcome)
        total_items = db.scalar(
            select(func.count()).select_from(IssueComparisonItemRow).where(*item_filter)
        ) or 0
        rows = list(
            db.scalars(
                select(IssueComparisonItemRow)
                .where(*item_filter)
                .order_by(IssueComparisonItemRow.outcome, IssueComparisonItemRow.id)
                .offset(offset)
                .limit(limit)
            ).all()
        )
        issue_ids = {
            issue_id
            for row in rows
            for issue_id in (row.baseline_issue_id, row.target_issue_id)
            if issue_id is not None
        }
        issues = (
            {
                row.id: row
                for row in db.scalars(
                    select(IssueRow).where(
                        IssueRow.id.in_(issue_ids),
                        IssueRow.run_id.in_(
                            (comparison.baseline_run_id, comparison.target_run_id)
                        ),
                    )
                ).all()
            }
            if issue_ids
            else {}
        )
        baseline_issues = {
            issue_id: issue
            for issue_id, issue in issues.items()
            if issue.run_id == comparison.baseline_run_id
        }
        target_issues = {
            issue_id: issue
            for issue_id, issue in issues.items()
            if issue.run_id == comparison.target_run_id
        }
        payload["items"] = [
            {
                "id": row.id,
                "outcome": row.outcome,
                "baseline_issue": _serialize_issue(
                    baseline_issues.get(row.baseline_issue_id)
                ),
                "target_issue": _serialize_issue(
                    target_issues.get(row.target_issue_id)
                ),
                "match_method": row.match_method,
                "match_score": row.match_score,
                "provenance": row.provenance,
                "baseline_latest_feedback": _comparison_feedback_snapshot(
                    row.provenance
                ),
            }
            for row in rows
        ]
        payload["page"]["returned"] = len(rows)
        payload["page"]["total"] = total_items
        payload["page"]["has_more"] = offset + len(rows) < total_items
        return payload


@app.get("/api/v1/analysis-runs/{run_id}/issues")
def get_issues(
    run_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> list[dict]:
    with SessionLocal() as db:
        if not _run_in_workspace(db, run_id, context.workspace_id):
            raise HTTPException(404, "分析任务不存在")
        rows = db.scalars(select(IssueRow).where(IssueRow.run_id == run_id)).all()
        return [_serialize_issue(row) for row in rows]


@app.get("/api/v1/analysis-runs/{run_id}/records")
def get_records(
    run_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    with SessionLocal() as db:
        run = _run_in_workspace(db, run_id, context.workspace_id)
        if not run:
            raise HTTPException(404, "分析任务不存在")
        rows = list(db.scalars(
            select(AnalysisRecordRow)
            .where(AnalysisRecordRow.run_id == run_id)
        ).all())
        rows.sort(key=record_sort_key)
        warnings = [
            row.message
            for row in db.scalars(
                select(RunEventRow).where(
                    RunEventRow.run_id == run_id, RunEventRow.stage == "warning"
                )
            ).all()
        ]
        records = [
            {"id": row.id, "kind": row.kind, "attrs": row.attrs, "evidence": row.evidence}
            for row in rows
        ]
        return {"records": records, "warnings": warnings, "record_count": len(records)}


@app.get(
    "/api/v1/analysis-runs/{run_id}/clarifications",
    response_model=list[SemanticReviewItemOut],
)
def get_clarifications(
    run_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> list[SemanticReviewItemOut]:
    """Return only reviewable questions/gaps, never arbitrary record attrs."""
    with SessionLocal() as db:
        run = _run_in_workspace(db, run_id, context.workspace_id)
        if not run:
            raise HTTPException(404, "分析任务不存在")
        if run.status != "completed":
            raise HTTPException(409, f"仅已完成任务可读取待澄清结果：{run.status}")
        rows = db.scalars(
            select(AnalysisRecordRow)
            .where(
                AnalysisRecordRow.run_id == run_id,
                AnalysisRecordRow.kind.in_(("clarification", "open_question")),
            )
            .order_by(AnalysisRecordRow.id)
        ).all()
        result: list[SemanticReviewItemOut] = []
        for row in rows:
            attrs = row.attrs or {}
            is_question = row.kind == "open_question"
            result.append(
                SemanticReviewItemOut(
                    id=row.id,
                    kind=row.kind,
                    category=None if is_question else attrs.get("category"),
                    modality=attrs.get(
                        "modality",
                        SemanticModality.interrogative.value
                        if is_question
                        else SemanticModality.uncertain.value,
                    ),
                    source_scope=attrs.get("source_scope", SourceScope.unknown.value),
                    certainty=attrs.get("certainty", CertaintyLevel.unknown.value),
                    text=attrs.get("question" if is_question else "summary", ""),
                    evidence=EvidenceSpan.model_validate(row.evidence),
                )
            )
        return result


def _completed_visualization_rows(db, run_id: str, workspace_id: str):
    run = _run_in_workspace(db, run_id, workspace_id)
    if not run:
        raise HTTPException(404, "分析任务不存在")
    if run.status != "completed":
        raise HTTPException(409, f"仅已完成任务可生成可视化：{run.status}")
    records = list(db.scalars(
        select(AnalysisRecordRow).where(AnalysisRecordRow.run_id == run_id)
    ).all())
    issues = list(db.scalars(
        select(IssueRow).where(IssueRow.run_id == run_id)
    ).all())
    return records, issues


@app.get("/api/v1/analysis-runs/{run_id}/graph", response_model=GraphResponse)
def get_graph(
    run_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> GraphResponse:
    with SessionLocal() as db:
        records, issues = _completed_visualization_rows(
            db, run_id, context.workspace_id
        )
        return project_graph(run_id, records, issues)


@app.get("/api/v1/analysis-runs/{run_id}/timeline", response_model=TimelineResponse)
def get_timeline(
    run_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> TimelineResponse:
    with SessionLocal() as db:
        records, issues = _completed_visualization_rows(
            db, run_id, context.workspace_id
        )
        return project_timeline(run_id, records, issues)


@app.get("/api/v1/analysis-runs/{run_id}/diagnostics")
def get_diagnostics(
    run_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    with SessionLocal() as db:
        if not _run_in_workspace(db, run_id, context.workspace_id):
            raise HTTPException(404, "分析任务不存在")
        row = db.get(AnalysisDiagnosticRow, run_id)
        return row.payload if row else {
            "chunking": {"total_chunks": 0, "documents": []},
            "aliases": {"declaration_count": 0, "trace_count": 0, "traces": []},
            "retrieval": {"candidate_count": 0, "consumed_count": 0, "traces": []},
        }


@app.get("/api/v1/analysis-runs/{run_id}/events")
async def stream_events(
    run_id: str,
    last_event_id: int = 0,
    last_event_id_header: Annotated[int | None, Header(alias="Last-Event-ID")] = None,
    context: AuthContext = Depends(get_auth_context),
):
    # Authorize before returning StreamingResponse so an unrelated run id is a
    # normal 404 and never establishes an SSE connection.
    with SessionLocal() as db:
        if not _run_in_workspace(db, run_id, context.workspace_id):
            raise HTTPException(404, "分析任务不存在")

    async def generate():
        cursor = max(last_event_id, last_event_id_header or 0)
        while True:
            with SessionLocal() as db:
                run = db.get(AnalysisRunRow, run_id)
                if not run:
                    yield "event: error\ndata: {\"message\":\"run not found\"}\n\n"; return
                rows = db.scalars(select(RunEventRow).where(RunEventRow.run_id == run_id, RunEventRow.id > cursor).order_by(RunEventRow.id)).all()
                for row in rows:
                    cursor = row.id
                    yield f"id: {row.id}\nevent: progress\ndata: {json.dumps({'stage': row.stage, 'progress': row.progress, 'message': row.message}, ensure_ascii=False)}\n\n"
                terminal = run.status in {"completed", "failed", "cancelled"}
                terminal_payload = {
                    "status": run.status,
                    "error": safe_persisted_analysis_error(run.error),
                }
            if terminal:
                yield f"event: terminal\ndata: {json.dumps(terminal_payload, ensure_ascii=False)}\n\n"
                return
            yield ": heartbeat\n\n"
            await asyncio.sleep(0.35)
    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/v1/issues/{issue_id}/feedback", status_code=201)
def feedback(
    issue_id: str,
    payload: FeedbackIn,
    response: Response,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    if payload.label not in {"accepted", "false_positive", "resolved"}:
        raise HTTPException(422, "label 必须是 accepted、false_positive 或 resolved")
    with SessionLocal() as db:
        if not _issue_in_workspace(db, issue_id, context.workspace_id):
            raise HTTPException(404, "问题不存在")
        latest = db.scalar(
            select(FeedbackRow)
            .where(FeedbackRow.issue_id == issue_id)
            .order_by(FeedbackRow.created_at.desc())
            .limit(1)
        )
        if latest and latest.label == payload.label and latest.comment == payload.comment:
            response.status_code = 200
            count = db.scalar(
                select(func.count()).select_from(FeedbackRow).where(FeedbackRow.issue_id == issue_id)
            ) or 0
            return {
                "id": latest.id, "issue_id": issue_id, "label": latest.label,
                "comment": latest.comment, "created_at": latest.created_at,
                "history_count": count, "duplicate_ignored": True,
            }
        row = FeedbackRow(
            issue_id=issue_id,
            created_by_user_id=context.user_id,
            label=payload.label,
            comment=payload.comment,
        )
        db.add(row); db.commit()
        count = db.scalar(
            select(func.count()).select_from(FeedbackRow).where(FeedbackRow.issue_id == issue_id)
        ) or 0
        return {
            "id": row.id, "issue_id": issue_id, "label": row.label,
            "comment": row.comment, "created_at": row.created_at,
            "history_count": count, "duplicate_ignored": False,
        }


@app.get("/api/v1/issues/{issue_id}/feedback")
def feedback_history(
    issue_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    with SessionLocal() as db:
        if not _issue_in_workspace(db, issue_id, context.workspace_id):
            raise HTTPException(404, "问题不存在")
        rows = db.scalars(
            select(FeedbackRow)
            .where(FeedbackRow.issue_id == issue_id)
            .order_by(FeedbackRow.created_at.desc())
        ).all()
        history = [
            {"id": row.id, "label": row.label, "comment": row.comment, "created_at": row.created_at}
            for row in rows
        ]
        return {"issue_id": issue_id, "latest": history[0] if history else None, "history": history}


@app.get("/api/v1/evaluations/{evaluation_id}")
def evaluation(
    evaluation_id: str,
    _context: AuthContext = Depends(get_auth_context),
) -> dict:
    if evaluation_id not in {"baseline", "latest"}: raise HTTPException(404, "仅内置 baseline/latest 评测")
    return {
        "benchmark_kind": "rule-engine synthetic directive regression",
        "natural_language_evaluation": False,
        "warning": "显式 @directive 回归只验证规则接线，不代表自然文本准确率。",
        "metrics": run_evaluation().model_dump(),
    }


@app.get("/metrics")
def metrics():
    try:
        database_metrics = render_analysis_metrics(SessionLocal)
    except AnalysisMetricsUnavailable:
        body = (
            "# HELP loreguard_analysis_metrics_database_available "
            "Whether this analysis metrics scrape was derived from the database.\n"
            "# TYPE loreguard_analysis_metrics_database_available gauge\n"
            "loreguard_analysis_metrics_database_available 0\n"
        )
        return Response(
            content=body,
            status_code=503,
            media_type=CONTENT_TYPE_LATEST,
        )
    return Response(
        content=generate_latest() + database_metrics,
        media_type=CONTENT_TYPE_LATEST,
    )


# Production build can be experienced with one Python process. API routes are
# registered first, then the single-page app handles every remaining path.
frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", SpaStaticFiles(directory=frontend_dist, html=True), name="web")
