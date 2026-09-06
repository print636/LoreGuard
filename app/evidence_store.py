from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Callable

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from .db import EmbeddingProfileRow, EvidenceChunkRow, EvidenceEmbeddingRow
from .embeddings import EmbeddingProfile
from .evidence_chunks import EvidenceChunk, EvidenceMatch, SnapshotDocumentKey


MAX_STORE_BATCH = 256


class EvidenceStoreError(ValueError):
    pass


class VectorSearchUnavailable(EvidenceStoreError):
    """Raised when exact pgvector search is unavailable for this database."""


@dataclass(frozen=True)
class EmbeddingCoverage:
    expected_count: int
    present_count: int
    missing_chunk_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.expected_count == self.present_count and not self.missing_chunk_ids


class SqlAlchemyEvidenceEmbeddingIndex:
    """Exact cosine search over one immutable snapshot/profile/chunker scope."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        if not callable(session_factory):
            raise TypeError("session factory must be callable")
        self._session_factory = session_factory

    def nearest(
        self,
        *,
        snapshots: Sequence[SnapshotDocumentKey],
        profile_id: str,
        chunker_version: str,
        query_vector: Sequence[float],
        limit: int,
    ) -> tuple[EvidenceMatch, ...]:
        prepared_snapshots = _validate_snapshot_filter(snapshots, allow_empty=False)
        if (
            not isinstance(profile_id, str)
            or len(profile_id) != 68
            or not profile_id.startswith("emb-")
        ):
            raise EvidenceStoreError("embedding profile id is invalid")
        if (
            not isinstance(chunker_version, str)
            or chunker_version != chunker_version.strip()
            or not chunker_version
            or len(chunker_version) > 80
        ):
            raise EvidenceStoreError("chunker version is invalid")
        if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= 50):
            raise EvidenceStoreError("vector search limit is invalid")

        with self._session_factory() as session:
            bind = session.get_bind()
            if bind.dialect.name != "postgresql":
                raise VectorSearchUnavailable("vector search requires PostgreSQL")
            with session.no_autoflush:
                profile_row = session.get(EmbeddingProfileRow, profile_id)
            if profile_row is None:
                raise EvidenceStoreError("embedding profile was not found")
            profile = _domain_profile(profile_row)
            prepared_vector = _validate_vector(
                query_vector, profile.dimensions, profile.normalized
            )
            clauses = [_snapshot_clause(snapshot) for snapshot in prepared_snapshots]
            candidates = (
                select(
                    EvidenceChunkRow.id.label("chunk_id"),
                    EvidenceChunkRow.project_id,
                    EvidenceChunkRow.document_id,
                    EvidenceChunkRow.document_version,
                    EvidenceChunkRow.content_sha256,
                    EvidenceChunkRow.chunker_version,
                    EvidenceChunkRow.ordinal,
                    EvidenceChunkRow.text,
                    EvidenceChunkRow.text_sha256,
                    EvidenceChunkRow.char_start,
                    EvidenceChunkRow.char_end,
                    EvidenceChunkRow.line_start,
                    EvidenceChunkRow.line_end,
                    EvidenceEmbeddingRow.vector,
                )
                .join(
                    EvidenceEmbeddingRow,
                    EvidenceEmbeddingRow.chunk_id == EvidenceChunkRow.id,
                )
                .where(
                    EvidenceEmbeddingRow.profile_id == profile_id,
                    EvidenceEmbeddingRow.dimensions == profile.dimensions,
                    EvidenceChunkRow.chunker_version == chunker_version,
                    or_(*clauses),
                )
                .cte("exact_evidence_candidates")
                .prefix_with("MATERIALIZED")
            )
            distance = candidates.c.vector.cosine_distance(
                list(prepared_vector)
            ).label("cosine_distance")
            with session.no_autoflush:
                rows = session.execute(
                    select(candidates, distance)
                    .order_by(distance.asc(), candidates.c.chunk_id.asc())
                    .limit(limit)
                ).mappings().all()

        matches: list[EvidenceMatch] = []
        for row in rows:
            try:
                distance_value = float(row["cosine_distance"])
            except (TypeError, ValueError, OverflowError):
                raise EvidenceStoreError("stored vector distance is invalid") from None
            if not math.isfinite(distance_value):
                raise EvidenceStoreError("stored vector distance is invalid")
            chunk = _domain_chunk_from_mapping(row)
            matches.append(EvidenceMatch(chunk=chunk, score=1.0 - distance_value))
        return tuple(matches)


def ensure_embedding_profile(session: Session, profile: EmbeddingProfile) -> None:
    """Validate and stage one profile without triggering an autoflush."""

    _validate_profile(profile)
    with session.no_autoflush:
        exists = _preflight_profile(session, profile)
    if not exists:
        session.add(_profile_row(profile))


def store_chunks(session: Session, chunks: Sequence[EvidenceChunk]) -> None:
    """Read-only preflight the entire batch, then stage all missing chunks."""

    prepared = _validate_chunk_batch(chunks, allow_empty=True)
    if not prepared:
        return
    with session.no_autoflush:
        existing_ids = _preflight_chunks(session, prepared)
    session.add_all(
        [_chunk_row(chunk) for chunk in prepared if chunk.chunk_id not in existing_ids]
    )


def store_embeddings(
    session: Session,
    *,
    profile: EmbeddingProfile,
    chunks: Sequence[EvidenceChunk],
    vectors: Sequence[Sequence[float]],
) -> None:
    """Preflight an entire embedding batch before any Session mutation.

    Application-level validation and identity collisions are detected before
    ``add``, ``flush`` or autoflush. A caller may therefore catch
    ``EvidenceStoreError`` and commit unrelated work without persisting a
    partial profile/chunk/vector batch.
    """

    _validate_profile(profile)
    prepared_chunks = _validate_chunk_batch(chunks, allow_empty=False)
    try:
        raw_vectors = tuple(vectors)
    except TypeError:
        raise EvidenceStoreError("embedding vectors are invalid") from None
    if len(prepared_chunks) != len(raw_vectors):
        raise EvidenceStoreError("chunk and vector counts do not match")
    prepared_vectors = tuple(
        _validate_vector(vector, profile.dimensions, profile.normalized)
        for vector in raw_vectors
    )

    # No helper in this block mutates or flushes the Session.
    with session.no_autoflush:
        profile_exists = _preflight_profile(session, profile)
        existing_chunk_ids = _preflight_chunks(session, prepared_chunks)
        existing_embedding_keys = _preflight_embeddings(
            session,
            profile=profile,
            chunks=prepared_chunks,
            vectors=prepared_vectors,
        )

    staged: list[object] = []
    if not profile_exists:
        staged.append(_profile_row(profile))
    staged.extend(
        _chunk_row(chunk)
        for chunk in prepared_chunks
        if chunk.chunk_id not in existing_chunk_ids
    )
    staged.extend(
        EvidenceEmbeddingRow(
            chunk_id=chunk.chunk_id,
            profile_id=profile.profile_id,
            dimensions=profile.dimensions,
            vector=list(vector),
        )
        for chunk, vector in zip(prepared_chunks, prepared_vectors, strict=True)
        if (chunk.chunk_id, profile.profile_id) not in existing_embedding_keys
    )
    session.add_all(staged)


def embedding_coverage(
    session: Session,
    *,
    profile: EmbeddingProfile,
    chunks: Sequence[EvidenceChunk],
) -> EmbeddingCoverage:
    """Verify durable, exact and usable vectors for one bounded chunk batch."""

    _validate_profile(profile)
    prepared = _validate_chunk_batch(chunks, allow_empty=True)
    if not prepared:
        return EmbeddingCoverage(0, 0, ())
    with session.no_autoflush:
        profile_exists = _preflight_profile(session, profile)
        existing_chunk_ids = _preflight_chunks(session, prepared)
        if not profile_exists:
            present_ids: set[str] = set()
        else:
            rows = session.scalars(
                select(EvidenceEmbeddingRow).where(
                    EvidenceEmbeddingRow.profile_id == profile.profile_id,
                    EvidenceEmbeddingRow.chunk_id.in_(
                        tuple(chunk.chunk_id for chunk in prepared)
                    ),
                )
            ).all()
            present_ids = set()
            for row in rows:
                if row.chunk_id not in existing_chunk_ids:
                    raise EvidenceStoreError("stored embedding has no exact chunk")
                if row.dimensions != profile.dimensions:
                    raise EvidenceStoreError("stored embedding dimension is invalid")
                try:
                    _validate_stored_vector(
                        row.vector, profile.dimensions, profile.normalized
                    )
                except EvidenceStoreError:
                    raise EvidenceStoreError("stored embedding vector is invalid") from None
                present_ids.add(row.chunk_id)
    missing_ids = tuple(
        chunk.chunk_id for chunk in prepared if chunk.chunk_id not in present_ids
    )
    return EmbeddingCoverage(
        expected_count=len(prepared),
        present_count=len(prepared) - len(missing_ids),
        missing_chunk_ids=missing_ids,
    )


def missing_embedding_chunks(
    session: Session,
    *,
    profile: EmbeddingProfile,
    chunks: Sequence[EvidenceChunk],
) -> tuple[EvidenceChunk, ...]:
    prepared = _validate_chunk_batch(chunks, allow_empty=True)
    coverage = embedding_coverage(session, profile=profile, chunks=prepared)
    missing = set(coverage.missing_chunk_ids)
    return tuple(chunk for chunk in prepared if chunk.chunk_id in missing)


def list_chunks_for_snapshots(
    session: Session, snapshots: Sequence[SnapshotDocumentKey]
) -> tuple[EvidenceChunk, ...]:
    """Return only chunks matching each exact immutable run input identity."""

    prepared = _validate_snapshot_filter(snapshots, allow_empty=True)
    if not prepared:
        return ()
    clauses = [_snapshot_clause(snapshot) for snapshot in prepared]
    with session.no_autoflush:
        rows = session.scalars(
            select(EvidenceChunkRow)
            .where(or_(*clauses))
            .order_by(
                EvidenceChunkRow.project_id,
                EvidenceChunkRow.document_id,
                EvidenceChunkRow.document_version,
                EvidenceChunkRow.ordinal,
            )
        ).all()
    try:
        return tuple(_domain_chunk(row) for row in rows)
    except ValueError:
        raise EvidenceStoreError("stored evidence chunk is invalid") from None


def _preflight_profile(session: Session, profile: EmbeddingProfile) -> bool:
    candidates = [
        row
        for row in session.new
        if isinstance(row, EmbeddingProfileRow)
        and (
            row.id == profile.profile_id
            or _profile_identity(row) == _domain_profile_identity(profile)
        )
    ]
    candidates.extend(
        session.scalars(
            select(EmbeddingProfileRow).where(
                or_(
                    EmbeddingProfileRow.id == profile.profile_id,
                    and_(
                        EmbeddingProfileRow.provider_kind == profile.provider_kind,
                        EmbeddingProfileRow.provider_namespace
                        == profile.provider_namespace,
                        EmbeddingProfileRow.model_identifier == profile.model_identifier,
                        EmbeddingProfileRow.model_revision == profile.model_revision,
                        EmbeddingProfileRow.deployment_fingerprint
                        == profile.deployment_fingerprint,
                        EmbeddingProfileRow.document_transform_identity
                        == profile.document_transform_identity,
                        EmbeddingProfileRow.query_transform_identity
                        == profile.query_transform_identity,
                        EmbeddingProfileRow.dimensions == profile.dimensions,
                        EmbeddingProfileRow.normalized == profile.normalized,
                    ),
                )
            )
        ).all()
    )
    exact = False
    expected = _profile_values(profile)
    for row in candidates:
        if row.id != profile.profile_id or _profile_values_from_row(row) != expected:
            raise EvidenceStoreError("embedding profile identity collision")
        exact = True
    return exact


def _preflight_chunks(
    session: Session, chunks: tuple[EvidenceChunk, ...]
) -> set[str]:
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    by_identity = {_chunk_logical_identity(chunk): chunk for chunk in chunks}
    candidates = [
        row
        for row in session.new
        if isinstance(row, EvidenceChunkRow)
        and (row.id in by_id or _chunk_row_logical_identity(row) in by_identity)
    ]
    logical_clauses = [
        and_(
            EvidenceChunkRow.project_id == identity[0],
            EvidenceChunkRow.document_id == identity[1],
            EvidenceChunkRow.document_version == identity[2],
            EvidenceChunkRow.content_sha256 == identity[3],
            EvidenceChunkRow.chunker_version == identity[4],
            EvidenceChunkRow.ordinal == identity[5],
        )
        for identity in by_identity
    ]
    candidates.extend(
        session.scalars(
            select(EvidenceChunkRow).where(
                or_(EvidenceChunkRow.id.in_(tuple(by_id)), *logical_clauses)
            )
        ).all()
    )

    exact_ids: set[str] = set()
    for row in candidates:
        expected = by_id.get(row.id)
        logical_expected = by_identity.get(_chunk_row_logical_identity(row))
        if expected is None or logical_expected is None or expected is not logical_expected:
            raise EvidenceStoreError("evidence chunk identity collision")
        if _chunk_values_from_row(row) != _chunk_values(expected):
            raise EvidenceStoreError("evidence chunk identity collision")
        exact_ids.add(row.id)
    return exact_ids


def _preflight_embeddings(
    session: Session,
    *,
    profile: EmbeddingProfile,
    chunks: tuple[EvidenceChunk, ...],
    vectors: tuple[tuple[float, ...], ...],
) -> set[tuple[str, str]]:
    expected = {
        (chunk.chunk_id, profile.profile_id): vector
        for chunk, vector in zip(chunks, vectors, strict=True)
    }
    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
    candidates = [
        row
        for row in session.new
        if isinstance(row, EvidenceEmbeddingRow)
        and (row.chunk_id, row.profile_id) in expected
    ]
    candidates.extend(
        session.scalars(
            select(EvidenceEmbeddingRow).where(
                EvidenceEmbeddingRow.profile_id == profile.profile_id,
                EvidenceEmbeddingRow.chunk_id.in_(chunk_ids),
            )
        ).all()
    )
    exact_keys: set[tuple[str, str]] = set()
    for row in candidates:
        key = (row.chunk_id, row.profile_id)
        vector = expected[key]
        try:
            stored_vector = tuple(float(value) for value in row.vector)
        except (TypeError, ValueError, OverflowError):
            raise EvidenceStoreError("stored embedding vector is invalid") from None
        if row.dimensions != profile.dimensions or not _vectors_equal(
            stored_vector, vector
        ):
            raise EvidenceStoreError("evidence embedding identity collision")
        exact_keys.add(key)
    return exact_keys


def _validate_profile(profile: EmbeddingProfile) -> None:
    if not isinstance(profile, EmbeddingProfile):
        raise EvidenceStoreError("embedding profile is invalid")
    try:
        EmbeddingProfile(
            profile_id=profile.profile_id,
            provider_kind=profile.provider_kind,
            provider_namespace=profile.provider_namespace,
            model_identifier=profile.model_identifier,
            model_revision=profile.model_revision,
            deployment_fingerprint=profile.deployment_fingerprint,
            document_transform_identity=profile.document_transform_identity,
            query_transform_identity=profile.query_transform_identity,
            dimensions=profile.dimensions,
            normalized=profile.normalized,
        )
    except (TypeError, ValueError):
        raise EvidenceStoreError("embedding profile is invalid") from None


def _validate_snapshot(snapshot: SnapshotDocumentKey) -> None:
    if not isinstance(snapshot, SnapshotDocumentKey):
        raise EvidenceStoreError("snapshot identity is invalid")
    try:
        SnapshotDocumentKey(
            project_id=snapshot.project_id,
            document_id=snapshot.document_id,
            document_version=snapshot.document_version,
            content_sha256=snapshot.content_sha256,
        )
    except (TypeError, ValueError):
        raise EvidenceStoreError("snapshot identity is invalid") from None


def _validate_snapshot_filter(
    snapshots: Sequence[SnapshotDocumentKey], *, allow_empty: bool
) -> tuple[SnapshotDocumentKey, ...]:
    try:
        prepared = tuple(snapshots)
    except TypeError:
        raise EvidenceStoreError("snapshot identities are invalid") from None
    if not prepared and not allow_empty:
        raise EvidenceStoreError("snapshot filter must not be empty")
    if len(prepared) > MAX_STORE_BATCH:
        raise EvidenceStoreError("snapshot filter limit was exceeded")
    for snapshot in prepared:
        _validate_snapshot(snapshot)
    if len(set(prepared)) != len(prepared):
        raise EvidenceStoreError("snapshot filter contains duplicate identities")
    return prepared


def _snapshot_clause(snapshot: SnapshotDocumentKey):
    return and_(
        EvidenceChunkRow.project_id == snapshot.project_id,
        EvidenceChunkRow.document_id == snapshot.document_id,
        EvidenceChunkRow.document_version == snapshot.document_version,
        EvidenceChunkRow.content_sha256 == snapshot.content_sha256,
    )


def _validate_chunk(chunk: EvidenceChunk) -> None:
    if not isinstance(chunk, EvidenceChunk):
        raise EvidenceStoreError("evidence chunk is invalid")
    _validate_snapshot(chunk.snapshot)
    try:
        EvidenceChunk(
            chunk_id=chunk.chunk_id,
            snapshot=chunk.snapshot,
            chunker_version=chunk.chunker_version,
            ordinal=chunk.ordinal,
            text=chunk.text,
            text_sha256=chunk.text_sha256,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
        )
    except (TypeError, ValueError):
        raise EvidenceStoreError("evidence chunk is invalid") from None


def _validate_chunk_batch(
    chunks: Sequence[EvidenceChunk], *, allow_empty: bool
) -> tuple[EvidenceChunk, ...]:
    try:
        prepared = tuple(chunks)
    except TypeError:
        raise EvidenceStoreError("evidence chunks are invalid") from None
    if not prepared and not allow_empty:
        raise EvidenceStoreError("embedding batch must not be empty")
    if len(prepared) > MAX_STORE_BATCH:
        raise EvidenceStoreError("evidence chunk batch limit was exceeded")
    for chunk in prepared:
        _validate_chunk(chunk)
    ids = [chunk.chunk_id for chunk in prepared]
    identities = [_chunk_logical_identity(chunk) for chunk in prepared]
    if len(set(ids)) != len(ids) or len(set(identities)) != len(identities):
        raise EvidenceStoreError("evidence chunk batch contains duplicate identities")
    return prepared


def _validate_vector(
    vector: Sequence[float], dimensions: int, normalized: bool
) -> tuple[float, ...]:
    try:
        if isinstance(vector, (str, bytes)) or len(vector) != dimensions:
            raise EvidenceStoreError("embedding vector dimension is invalid")
    except TypeError:
        raise EvidenceStoreError("embedding vector is invalid") from None
    converted: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvidenceStoreError("embedding vector value is invalid")
        try:
            number = float(value)
        except (OverflowError, ValueError):
            raise EvidenceStoreError("embedding vector value is invalid") from None
        if not math.isfinite(number):
            raise EvidenceStoreError("embedding vector value is invalid")
        converted.append(number)
    norm = math.sqrt(sum(value * value for value in converted))
    if not math.isfinite(norm) or norm <= 0:
        raise EvidenceStoreError("embedding vector norm is invalid")
    if normalized and not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise EvidenceStoreError("embedding vector normalization is invalid")
    return tuple(converted)


def _validate_stored_vector(
    vector: Sequence[float], dimensions: int, normalized: bool
) -> tuple[float, ...]:
    try:
        if isinstance(vector, (str, bytes)) or len(vector) != dimensions:
            raise EvidenceStoreError("stored embedding vector dimension is invalid")
    except TypeError:
        raise EvidenceStoreError("stored embedding vector is invalid") from None
    converted: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise EvidenceStoreError("stored embedding vector value is invalid")
        try:
            number = float(value)
        except (OverflowError, ValueError):
            raise EvidenceStoreError("stored embedding vector value is invalid") from None
        if not math.isfinite(number):
            raise EvidenceStoreError("stored embedding vector value is invalid")
        converted.append(number)
    norm = math.sqrt(sum(value * value for value in converted))
    if not math.isfinite(norm) or norm <= 0:
        raise EvidenceStoreError("stored embedding vector norm is invalid")
    if normalized and not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise EvidenceStoreError("stored embedding vector normalization is invalid")
    return tuple(converted)


def _vectors_equal(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-7)
        for a, b in zip(left, right, strict=True)
    )


def _profile_row(profile: EmbeddingProfile) -> EmbeddingProfileRow:
    return EmbeddingProfileRow(
        id=profile.profile_id,
        provider_kind=profile.provider_kind,
        provider_namespace=profile.provider_namespace,
        model_identifier=profile.model_identifier,
        model_revision=profile.model_revision,
        deployment_fingerprint=profile.deployment_fingerprint,
        document_transform_identity=profile.document_transform_identity,
        query_transform_identity=profile.query_transform_identity,
        dimensions=profile.dimensions,
        normalized=profile.normalized,
    )


def _chunk_row(chunk: EvidenceChunk) -> EvidenceChunkRow:
    return EvidenceChunkRow(
        id=chunk.chunk_id,
        project_id=chunk.snapshot.project_id,
        document_id=chunk.snapshot.document_id,
        document_version=chunk.snapshot.document_version,
        content_sha256=chunk.snapshot.content_sha256,
        chunker_version=chunk.chunker_version,
        ordinal=chunk.ordinal,
        text=chunk.text,
        text_sha256=chunk.text_sha256,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        line_start=chunk.line_start,
        line_end=chunk.line_end,
    )


def _profile_values(profile: EmbeddingProfile) -> tuple[object, ...]:
    return (profile.profile_id, *_domain_profile_identity(profile))


def _profile_values_from_row(row: EmbeddingProfileRow) -> tuple[object, ...]:
    return (row.id, *_profile_identity(row))


def _domain_profile_identity(profile: EmbeddingProfile) -> tuple[object, ...]:
    return (
        profile.provider_kind,
        profile.provider_namespace,
        profile.model_identifier,
        profile.model_revision,
        profile.deployment_fingerprint,
        profile.document_transform_identity,
        profile.query_transform_identity,
        profile.dimensions,
        profile.normalized,
    )


def _profile_identity(row: EmbeddingProfileRow) -> tuple[object, ...]:
    return (
        row.provider_kind,
        row.provider_namespace,
        row.model_identifier,
        row.model_revision,
        row.deployment_fingerprint,
        row.document_transform_identity,
        row.query_transform_identity,
        row.dimensions,
        row.normalized,
    )


def _chunk_logical_identity(chunk: EvidenceChunk) -> tuple[object, ...]:
    return (
        chunk.snapshot.project_id,
        chunk.snapshot.document_id,
        chunk.snapshot.document_version,
        chunk.snapshot.content_sha256,
        chunk.chunker_version,
        chunk.ordinal,
    )


def _chunk_row_logical_identity(row: EvidenceChunkRow) -> tuple[object, ...]:
    return (
        row.project_id,
        row.document_id,
        row.document_version,
        row.content_sha256,
        row.chunker_version,
        row.ordinal,
    )


def _chunk_values(chunk: EvidenceChunk) -> tuple[object, ...]:
    return (
        *_chunk_logical_identity(chunk),
        chunk.text,
        chunk.text_sha256,
        chunk.char_start,
        chunk.char_end,
        chunk.line_start,
        chunk.line_end,
    )


def _chunk_values_from_row(row: EvidenceChunkRow) -> tuple[object, ...]:
    return (
        *_chunk_row_logical_identity(row),
        row.text,
        row.text_sha256,
        row.char_start,
        row.char_end,
        row.line_start,
        row.line_end,
    )


def _domain_chunk(row: EvidenceChunkRow) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=row.id,
        snapshot=SnapshotDocumentKey(
            project_id=row.project_id,
            document_id=row.document_id,
            document_version=row.document_version,
            content_sha256=row.content_sha256,
        ),
        chunker_version=row.chunker_version,
        ordinal=row.ordinal,
        text=row.text,
        text_sha256=row.text_sha256,
        char_start=row.char_start,
        char_end=row.char_end,
        line_start=row.line_start,
        line_end=row.line_end,
    )


def _domain_chunk_from_mapping(row) -> EvidenceChunk:
    try:
        return EvidenceChunk(
            chunk_id=row["chunk_id"],
            snapshot=SnapshotDocumentKey(
                project_id=row["project_id"],
                document_id=row["document_id"],
                document_version=row["document_version"],
                content_sha256=row["content_sha256"],
            ),
            chunker_version=row["chunker_version"],
            ordinal=row["ordinal"],
            text=row["text"],
            text_sha256=row["text_sha256"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            line_start=row["line_start"],
            line_end=row["line_end"],
        )
    except (KeyError, TypeError, ValueError):
        raise EvidenceStoreError("stored evidence chunk is invalid") from None


def _domain_profile(row: EmbeddingProfileRow) -> EmbeddingProfile:
    try:
        return EmbeddingProfile(
            profile_id=row.id,
            provider_kind=row.provider_kind,
            provider_namespace=row.provider_namespace,
            model_identifier=row.model_identifier,
            model_revision=row.model_revision,
            deployment_fingerprint=row.deployment_fingerprint,
            document_transform_identity=row.document_transform_identity,
            query_transform_identity=row.query_transform_identity,
            dimensions=row.dimensions,
            normalized=row.normalized,
        )
    except (TypeError, ValueError):
        raise EvidenceStoreError("stored embedding profile is invalid") from None
