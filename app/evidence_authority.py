from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .evidence_chunks import EvidenceChunk, SnapshotDocumentKey
from .evidence_investigator import (
    HASH_PATTERN,
    SEED_REF_PATTERN,
    CandidateRecordSubmission,
    InvestigationSeed,
    InvestigatorRejected,
    ReadSpanArgs,
    clone_investigation_seed,
)


_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


@dataclass(frozen=True, slots=True)
class ScopedEvidenceDocument:
    """One immutable document version admitted to an investigation run."""

    snapshot: SnapshotDocumentKey
    content: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.snapshot) is not SnapshotDocumentKey:
            raise ValueError("scope snapshot is invalid")
        if type(self.content) is not str:
            raise ValueError("scope document content is invalid")
        if _sha256(self.content) != self.snapshot.content_sha256:
            raise ValueError("scope document hash mismatch")

    @property
    def normalized_content(self) -> str:
        return self.content.replace("\r\n", "\n").replace("\r", "\n")

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(self.normalized_content.splitlines())


@dataclass(frozen=True, slots=True)
class InvestigationScope:
    """Closed project/document snapshot; model arguments cannot enlarge it."""

    run_hash: str
    project_id: str
    documents: tuple[ScopedEvidenceDocument, ...] = field(repr=False)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        project_id: str,
        documents: Sequence[ScopedEvidenceDocument],
    ) -> InvestigationScope:
        if not isinstance(run_id, str) or not run_id.strip() or len(run_id) > 256:
            raise ValueError("scope run id is invalid")
        try:
            prepared = tuple(documents)
        except TypeError:
            raise ValueError("scope documents are invalid") from None
        return cls(
            run_hash=_sha256(run_id),
            project_id=project_id,
            documents=prepared,
        )

    def __post_init__(self) -> None:
        if type(self.run_hash) is not str or HASH_PATTERN.fullmatch(self.run_hash) is None:
            raise ValueError("scope run hash is invalid")
        if (
            type(self.project_id) is not str
            or not self.project_id
            or self.project_id != self.project_id.strip()
            or len(self.project_id) > 36
        ):
            raise ValueError("scope project id is invalid")
        if type(self.documents) is not tuple or not 1 <= len(self.documents) <= 256:
            raise ValueError("scope document count is invalid")

        snapshots: set[SnapshotDocumentKey] = set()
        document_ids: set[str] = set()
        for document in self.documents:
            if type(document) is not ScopedEvidenceDocument:
                raise ValueError("scope document is invalid")
            if document.snapshot.project_id != self.project_id:
                raise ValueError("scope contains another project")
            if (
                document.snapshot in snapshots
                or document.snapshot.document_id in document_ids
            ):
                raise ValueError("scope contains duplicate document versions")
            snapshots.add(document.snapshot)
            document_ids.add(document.snapshot.document_id)


def clone_investigation_scope(scope: InvestigationScope) -> InvestigationScope:
    """Rebuild an exact immutable snapshot without retaining caller aliases."""

    if type(scope) is not InvestigationScope:
        raise ValueError("scope is invalid")
    try:
        if type(scope.documents) is not tuple:
            raise ValueError("scope is invalid")
        documents = tuple(
            ScopedEvidenceDocument(
                snapshot=_copy_snapshot(document.snapshot),
                content=document.content,
            )
            for document in scope.documents
            if type(document) is ScopedEvidenceDocument
        )
        if len(documents) != len(scope.documents):
            raise ValueError("scope document is invalid")
        return InvestigationScope(
            run_hash=scope.run_hash,
            project_id=scope.project_id,
            documents=documents,
        )
    except (AttributeError, TypeError, ValueError):
        raise ValueError("scope is invalid") from None


def clone_evidence_chunks(
    chunks: Sequence[EvidenceChunk],
) -> tuple[EvidenceChunk, ...]:
    """Atomically reconstruct an untrusted retrieval batch before field access."""

    try:
        prepared = tuple(chunks)
    except (AttributeError, TypeError, ValueError):
        raise InvestigatorRejected("snapshot_mismatch") from None
    if any(type(row) is not EvidenceChunk for row in prepared):
        raise InvestigatorRejected("snapshot_mismatch")
    try:
        cloned = tuple(_copy_evidence_chunk(row) for row in prepared)
    except (AttributeError, TypeError, ValueError):
        raise InvestigatorRejected("evidence_hash_mismatch") from None
    try:
        contains_duplicates = len({row.chunk_id for row in cloned}) != len(cloned)
    except (AttributeError, TypeError, ValueError):
        raise InvestigatorRejected("evidence_hash_mismatch") from None
    if contains_duplicates:
        raise InvestigatorRejected("snapshot_mismatch")
    return cloned


@dataclass(frozen=True, slots=True)
class SearchGrant:
    result_ref: str
    run_hash: str
    seed_ref: str
    snapshot: SnapshotDocumentKey = field(repr=False)
    chunk_id: str = field(repr=False)
    text_sha256: str = field(repr=False)
    char_start: int = field(repr=False)
    char_end: int = field(repr=False)
    line_start: int
    line_end: int


@dataclass(frozen=True, slots=True)
class SpanGrant:
    span_ref: str
    run_hash: str
    seed_ref: str
    result_ref: str
    snapshot: SnapshotDocumentKey = field(repr=False)
    char_start: int = field(repr=False)
    char_end: int = field(repr=False)
    line_start: int
    line_end: int
    text_sha256: str = field(repr=False)
    text: str = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class SearchObservation:
    result_ref: str
    line_start: int
    line_end: int


@dataclass(frozen=True, slots=True)
class ReadObservation:
    span_ref: str
    line_start: int
    line_end: int
    text: str = field(repr=False)


class EvidenceGrantAuthority:
    """Mint opaque capabilities only after exact snapshot verification."""

    def __init__(
        self,
        *,
        scope: InvestigationScope,
        seeds: Iterable[InvestigationSeed],
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        if type(scope) is not InvestigationScope:
            raise TypeError("scope is invalid")
        try:
            scope = clone_investigation_scope(scope)
        except ValueError:
            raise ValueError("scope is invalid") from None
        try:
            supplied_seeds = tuple(seeds)
        except TypeError:
            raise ValueError("authority seeds are invalid") from None
        try:
            prepared_seeds = tuple(
                clone_investigation_seed(seed) for seed in supplied_seeds
            )
        except ValueError:
            raise ValueError("authority seeds are invalid") from None
        if not prepared_seeds or any(
            re.fullmatch(SEED_REF_PATTERN, seed.seed_ref) is None
            or seed.run_hash != scope.run_hash
            for seed in prepared_seeds
        ):
            raise ValueError("authority seeds are invalid")
        if len({seed.seed_ref for seed in prepared_seeds}) != len(prepared_seeds):
            raise ValueError("authority seeds contain duplicates")
        self._scope = scope
        self._seed_candidate_kinds = {
            seed.seed_ref: frozenset(seed.allowed_candidate_kinds)
            for seed in prepared_seeds
        }
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(18))
        if not callable(self._token_factory):
            raise TypeError("token factory must be callable")
        self._documents = {row.snapshot: row for row in scope.documents}
        self._search_grants: dict[str, SearchGrant] = {}
        self._span_grants: dict[str, SpanGrant] = {}

    def issue_search_grants(
        self, seed_ref: str, chunks: Sequence[EvidenceChunk]
    ) -> tuple[SearchGrant, ...]:
        self._require_seed(seed_ref)
        prepared = clone_evidence_chunks(chunks)

        # Atomic preflight: do not mint any reference until the whole batch passes.
        for chunk in prepared:
            self._validate_chunk(chunk)
        refs = self._new_refs("result", len(prepared), self._search_grants)
        grants = tuple(
            SearchGrant(
                result_ref=result_ref,
                run_hash=self._scope.run_hash,
                seed_ref=seed_ref,
                snapshot=chunk.snapshot,
                chunk_id=chunk.chunk_id,
                text_sha256=chunk.text_sha256,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
            )
            for result_ref, chunk in zip(refs, prepared, strict=True)
        )
        self._search_grants.update({row.result_ref: row for row in grants})
        return tuple(_copy_search_grant(row) for row in grants)

    def read_span(self, args: ReadSpanArgs, *, max_read_lines: int) -> SpanGrant:
        """Validate and mint a read capability as one indivisible operation."""

        if type(args) is not ReadSpanArgs:
            raise InvestigatorRejected("invalid_tool_arguments")
        try:
            args = _clone_model(args, ReadSpanArgs)
        except (AttributeError, TypeError, ValueError):
            raise InvestigatorRejected("invalid_tool_arguments") from None
        if (
            type(max_read_lines) is not int
            or not 1 <= max_read_lines <= 20
        ):
            raise InvestigatorRejected("evidence_range")
        self._require_seed(args.seed_ref)
        result = self._search_grants.get(args.result_ref)
        if result is None:
            raise InvestigatorRejected("unknown_result_ref")
        if result.run_hash != self._scope.run_hash:
            raise InvestigatorRejected("cross_run")
        if result.seed_ref != args.seed_ref:
            raise InvestigatorRejected("cross_seed")
        if (
            args.line_start < result.line_start
            or args.line_end > result.line_end
            or args.line_end - args.line_start + 1 > max_read_lines
        ):
            raise InvestigatorRejected("evidence_range")
        document = self._documents.get(result.snapshot)
        if document is None:
            raise InvestigatorRejected("snapshot_mismatch")
        chunk_text = document.normalized_content[result.char_start : result.char_end]
        if (
            result.char_start < 0
            or result.char_end <= result.char_start
            or result.char_end > len(document.normalized_content)
            or _sha256(chunk_text) != result.text_sha256
        ):
            raise InvestigatorRejected("evidence_hash_mismatch")
        lines = document.lines
        if args.line_end > len(lines):
            raise InvestigatorRejected("evidence_range")
        requested_start, requested_end = _line_character_range(
            document.normalized_content,
            args.line_start,
            args.line_end,
        )
        char_start = max(requested_start, result.char_start)
        char_end = min(requested_end, result.char_end)
        if char_end <= char_start:
            raise InvestigatorRejected("evidence_range")
        text = document.normalized_content[char_start:char_end]
        if not text:
            raise InvestigatorRejected("evidence_range")
        line_start = document.normalized_content[:char_start].count("\n") + 1
        line_end = line_start + text[:-1].count("\n")
        span_ref = self._new_refs("span", 1, self._span_grants)[0]
        grant = SpanGrant(
            span_ref=span_ref,
            run_hash=self._scope.run_hash,
            seed_ref=args.seed_ref,
            result_ref=args.result_ref,
            snapshot=result.snapshot,
            char_start=char_start,
            char_end=char_end,
            line_start=line_start,
            line_end=line_end,
            text_sha256=_sha256(text),
            text=text,
        )
        self._span_grants[span_ref] = grant
        return _copy_span_grant(grant)

    def authorize_candidate(
        self, seed_ref: str, candidate: CandidateRecordSubmission
    ) -> SpanGrant:
        if type(candidate) is not CandidateRecordSubmission:
            raise InvestigatorRejected("invalid_tool_arguments")
        try:
            candidate = _clone_model(candidate, CandidateRecordSubmission)
        except (AttributeError, TypeError, ValueError):
            raise InvestigatorRejected("invalid_tool_arguments") from None
        self._require_seed(seed_ref)
        if candidate.kind not in self._seed_candidate_kinds[seed_ref]:
            raise InvestigatorRejected("candidate_kind_forbidden")
        grant = self._span_grants.get(candidate.span_ref)
        if grant is None:
            raise InvestigatorRejected("unknown_span_ref")
        if grant.run_hash != self._scope.run_hash:
            raise InvestigatorRejected("cross_run")
        if grant.seed_ref != seed_ref:
            raise InvestigatorRejected("cross_seed")
        if (
            candidate.source_line_start < grant.line_start
            or candidate.source_line_end > grant.line_end
        ):
            raise InvestigatorRejected("evidence_range")
        document = self._documents.get(grant.snapshot)
        if document is None:
            raise InvestigatorRejected("snapshot_mismatch")
        text = document.normalized_content[grant.char_start : grant.char_end]
        if text != grant.text or _sha256(text) != grant.text_sha256:
            raise InvestigatorRejected("evidence_hash_mismatch")
        return _copy_span_grant(grant)

    def _validate_chunk(self, chunk: EvidenceChunk) -> None:
        document = self._documents.get(chunk.snapshot)
        if document is None:
            raise InvestigatorRejected("snapshot_mismatch")
        normalized = document.normalized_content
        expected_line_start = normalized[: chunk.char_start].count("\n") + 1
        expected_line_end = expected_line_start + chunk.text[:-1].count("\n")
        if (
            chunk.char_end > len(normalized)
            or normalized[chunk.char_start : chunk.char_end] != chunk.text
            or _sha256(chunk.text) != chunk.text_sha256
            or chunk.line_start != expected_line_start
            or chunk.line_end != expected_line_end
        ):
            raise InvestigatorRejected("evidence_hash_mismatch")

    def _require_seed(self, seed_ref: str) -> None:
        if not isinstance(seed_ref, str) or seed_ref not in self._seed_candidate_kinds:
            raise InvestigatorRejected("unknown_seed")

    def _new_refs(self, prefix: str, count: int, existing: dict[str, Any]) -> list[str]:
        refs: list[str] = []
        for _ in range(count):
            for _attempt in range(8):
                token = self._token_factory()
                if isinstance(token, str) and _TOKEN_PATTERN.fullmatch(token) is not None:
                    candidate = f"{prefix}_{token}"
                    if candidate not in existing and candidate not in refs:
                        refs.append(candidate)
                        break
            else:
                raise InvestigatorRejected("internal_failure")
        return refs


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _line_character_range(content: str, line_start: int, line_end: int) -> tuple[int, int]:
    starts = [0]
    starts.extend(index + 1 for index, character in enumerate(content) if character == "\n")
    if line_start < 1 or line_end < line_start or line_end > len(starts):
        raise InvestigatorRejected("evidence_range")
    char_start = starts[line_start - 1]
    char_end = starts[line_end] - 1 if line_end < len(starts) else len(content)
    return char_start, char_end


def _copy_snapshot(snapshot: SnapshotDocumentKey) -> SnapshotDocumentKey:
    if (
        type(snapshot) is not SnapshotDocumentKey
        or type(snapshot.project_id) is not str
        or type(snapshot.document_id) is not str
        or type(snapshot.document_version) is not int
        or type(snapshot.content_sha256) is not str
    ):
        raise ValueError("snapshot is invalid")
    return SnapshotDocumentKey(
        project_id=snapshot.project_id,
        document_id=snapshot.document_id,
        document_version=snapshot.document_version,
        content_sha256=snapshot.content_sha256,
    )


def _copy_evidence_chunk(chunk: EvidenceChunk) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk.chunk_id,
        snapshot=_copy_snapshot(chunk.snapshot),
        chunker_version=chunk.chunker_version,
        ordinal=chunk.ordinal,
        text=chunk.text,
        text_sha256=chunk.text_sha256,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        line_start=chunk.line_start,
        line_end=chunk.line_end,
    )


def _clone_model(value: Any, model: type[Any]) -> Any:
    payload = json.loads(
        json.dumps(
            value.model_dump(mode="json", warnings="error"),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    )
    return model.model_validate(payload)


def _copy_search_grant(grant: SearchGrant) -> SearchGrant:
    return SearchGrant(
        result_ref=grant.result_ref,
        run_hash=grant.run_hash,
        seed_ref=grant.seed_ref,
        snapshot=_copy_snapshot(grant.snapshot),
        chunk_id=grant.chunk_id,
        text_sha256=grant.text_sha256,
        char_start=grant.char_start,
        char_end=grant.char_end,
        line_start=grant.line_start,
        line_end=grant.line_end,
    )


def _copy_span_grant(grant: SpanGrant) -> SpanGrant:
    return SpanGrant(
        span_ref=grant.span_ref,
        run_hash=grant.run_hash,
        seed_ref=grant.seed_ref,
        result_ref=grant.result_ref,
        snapshot=_copy_snapshot(grant.snapshot),
        char_start=grant.char_start,
        char_end=grant.char_end,
        line_start=grant.line_start,
        line_end=grant.line_end,
        text_sha256=grant.text_sha256,
        text=grant.text,
    )
