from __future__ import annotations

import bisect
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class SnapshotDocumentKey:
    """Exact immutable document identity captured by an analysis run."""

    project_id: str
    document_id: str
    document_version: int
    content_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.project_id, str)
            or self.project_id != self.project_id.strip()
            or not self.project_id
            or len(self.project_id) > 36
        ):
            raise ValueError("snapshot project id is invalid")
        if (
            not isinstance(self.document_id, str)
            or self.document_id != self.document_id.strip()
            or not self.document_id
            or len(self.document_id) > 36
        ):
            raise ValueError("snapshot document id is invalid")
        if (
            isinstance(self.document_version, bool)
            or not isinstance(self.document_version, int)
            or self.document_version < 1
        ):
            raise ValueError("snapshot document version is invalid")
        if (
            not isinstance(self.content_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.content_sha256) is None
        ):
            raise ValueError("snapshot content hash is invalid")


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    snapshot: SnapshotDocumentKey
    chunker_version: str
    ordinal: int
    text: str
    text_sha256: str
    char_start: int
    char_end: int
    line_start: int
    line_end: int

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, SnapshotDocumentKey):
            raise ValueError("chunk snapshot is invalid")
        if (
            not isinstance(self.chunker_version, str)
            or self.chunker_version != self.chunker_version.strip()
            or not self.chunker_version
            or len(self.chunker_version) > 80
        ):
            raise ValueError("chunker version is invalid")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise ValueError("chunk ordinal is invalid")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("chunk text is invalid")
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.text_sha256:
            raise ValueError("chunk text hash is invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.char_start, self.char_end, self.line_start, self.line_end)
        ):
            raise ValueError("chunk range is invalid")
        if self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError("chunk character range is invalid")
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("chunk text length does not match its character range")
        if self.line_start < 1 or self.line_end < self.line_start:
            raise ValueError("chunk line range is invalid")
        if self.line_end != self.line_start + self.text[:-1].count("\n"):
            raise ValueError("chunk text does not match its line range")
        if self.chunk_id != _chunk_id(
            snapshot=self.snapshot,
            chunker_version=self.chunker_version,
            ordinal=self.ordinal,
            char_start=self.char_start,
            char_end=self.char_end,
            text_sha256=self.text_sha256,
        ):
            raise ValueError("chunk id is invalid")


@dataclass(frozen=True)
class EvidenceMatch:
    chunk: EvidenceChunk
    score: float


class EvidenceEmbeddingIndex(Protocol):
    """Future retrieval seam with complete immutable-index isolation.

    Implementations must match ``chunker_version`` by exact equality together
    with every snapshot field and ``profile_id``.  Requiring it here prevents a
    later index adapter from silently mixing chunks produced by two chunking
    configurations for the same document snapshot and embedding profile.
    """

    def nearest(
        self,
        *,
        snapshots: Sequence[SnapshotDocumentKey],
        profile_id: str,
        chunker_version: str,
        query_vector: Sequence[float],
        limit: int,
    ) -> Sequence[EvidenceMatch]:
        ...


class EvidenceChunker:
    """Deterministic paragraph/sentence chunker for Chinese narrative text.

    Offsets refer to the normalized (LF-only) content retained by the chunks.
    Line numbers are one-based and remain equivalent for CRLF input.
    """

    SENTENCE_ENDINGS = frozenset("。！？!?；;……")

    def __init__(
        self,
        *,
        target_chars: int = 450,
        min_chars: int = 300,
        max_chars: int = 600,
        overlap_chars: int = 80,
        version: str = "line-sentence-v1",
    ) -> None:
        sizes = (min_chars, target_chars, max_chars, overlap_chars)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in sizes):
            raise ValueError("chunk size bounds are invalid")
        if not (1 <= min_chars <= target_chars <= max_chars):
            raise ValueError("chunk size bounds are invalid")
        if not (0 <= overlap_chars < min_chars):
            raise ValueError("chunk overlap must be smaller than the minimum chunk size")
        if (
            not isinstance(version, str)
            or version != version.strip()
            or not version
            or len(version) > 60
        ):
            raise ValueError("chunker version is invalid")
        self.target_chars = target_chars
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        configuration = json.dumps(
            {
                "target_chars": target_chars,
                "min_chars": min_chars,
                "max_chars": max_chars,
                "overlap_chars": overlap_chars,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        configuration_hash = hashlib.sha256(configuration.encode("utf-8")).hexdigest()
        self.version = f"{version}@{configuration_hash[:12]}"

    def chunk(
        self,
        *,
        project_id: str,
        document_id: str,
        document_version: int,
        content: str,
        content_sha256: str | None = None,
    ) -> tuple[EvidenceChunk, ...]:
        if not isinstance(content, str):
            raise ValueError("document content must be text")
        if (
            not isinstance(project_id, str)
            or not project_id.strip()
            or len(project_id) > 36
            or not isinstance(document_id, str)
            or not document_id.strip()
            or len(document_id) > 36
        ):
            raise ValueError("snapshot identifiers are invalid")
        if (
            isinstance(document_version, bool)
            or not isinstance(document_version, int)
            or document_version < 1
        ):
            raise ValueError("document version must be positive")
        # Snapshot identity must match AnalysisRunInputRow, which hashes the
        # exact uploaded content.  Chunk offsets use the LF-normalized view,
        # but that normalization must never silently change snapshot identity.
        actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        if content_sha256 is not None:
            if (
                not isinstance(content_sha256, str)
                or content_sha256.lower() != actual_hash
            ):
                raise ValueError("snapshot content hash does not match uploaded content")
        snapshot = SnapshotDocumentKey(
            project_id=project_id,
            document_id=document_id,
            document_version=document_version,
            content_sha256=actual_hash,
        )
        if not normalized.strip():
            return ()

        boundaries = self._boundaries(normalized)
        newlines = [index for index, character in enumerate(normalized) if character == "\n"]
        chunks: list[EvidenceChunk] = []
        start = 0
        while start < len(normalized):
            end = self._choose_end(start, len(normalized), boundaries)
            if end <= start:
                raise RuntimeError("chunker failed to make forward progress")
            text_start = start
            text_end = end
            while text_start < text_end and normalized[text_start].isspace():
                text_start += 1
            while text_end > text_start and normalized[text_end - 1].isspace():
                text_end -= 1
            text = normalized[text_start:text_end]
            if text.strip():
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                ordinal = len(chunks)
                chunks.append(
                    EvidenceChunk(
                        chunk_id=_chunk_id(
                            snapshot=snapshot,
                            chunker_version=self.version,
                            ordinal=ordinal,
                            char_start=text_start,
                            char_end=text_end,
                            text_sha256=text_hash,
                        ),
                        snapshot=snapshot,
                        chunker_version=self.version,
                        ordinal=ordinal,
                        text=text,
                        text_sha256=text_hash,
                        char_start=text_start,
                        char_end=text_end,
                        line_start=self._line_number(newlines, text_start),
                        line_end=self._line_number(newlines, text_end - 1),
                    )
                )
            if end == len(normalized):
                break
            start = self._next_start(start, end, boundaries)
        return tuple(chunks)

    def _boundaries(self, text: str) -> tuple[int, ...]:
        values = {len(text)}
        for index, character in enumerate(text):
            if character == "\n" or character in self.SENTENCE_ENDINGS:
                values.add(index + 1)
        return tuple(sorted(values))

    def _choose_end(
        self, start: int, content_length: int, boundaries: tuple[int, ...]
    ) -> int:
        hard_end = min(content_length, start + self.max_chars)
        target = min(content_length, start + self.target_chars)
        low = bisect.bisect_left(boundaries, start + self.min_chars)
        high = bisect.bisect_right(boundaries, hard_end)
        candidates = boundaries[low:high]
        if candidates:
            at_or_after_target = [value for value in candidates if value >= target]
            return at_or_after_target[0] if at_or_after_target else candidates[-1]
        return hard_end

    def _next_start(
        self, previous_start: int, end: int, boundaries: tuple[int, ...]
    ) -> int:
        if self.overlap_chars == 0:
            return end
        earliest = max(previous_start + 1, end - self.overlap_chars)
        index = bisect.bisect_left(boundaries, earliest)
        if index < len(boundaries) and boundaries[index] < end:
            return boundaries[index]
        return end

    @staticmethod
    def _line_number(newlines: list[int], character_offset: int) -> int:
        return bisect.bisect_left(newlines, character_offset) + 1


def _chunk_id(
    *,
    snapshot: SnapshotDocumentKey,
    chunker_version: str,
    ordinal: int,
    char_start: int,
    char_end: int,
    text_sha256: str,
) -> str:
    identity = json.dumps(
        {
            "project_id": snapshot.project_id,
            "document_id": snapshot.document_id,
            "document_version": snapshot.document_version,
            "content_sha256": snapshot.content_sha256,
            "chunker_version": chunker_version,
            "ordinal": ordinal,
            "char_start": char_start,
            "char_end": char_end,
            "text_sha256": text_sha256,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"chk-{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"
