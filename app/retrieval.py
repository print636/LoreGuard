from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

from .domain import EvidenceSpan, ParsedDirective


def _tokens(text: str) -> list[str]:
    lowered = re.sub(r"\s+", "", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]", lowered)
    words = re.findall(r"[a-z0-9_]+", lowered)
    ngrams = [lowered[index : index + 2] for index in range(max(0, len(lowered) - 1))]
    return chinese + words + ngrams


def _bucket(token: str, dims: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % dims


def _hash_vector(text: str, dims: int = 128) -> list[float]:
    vector = [0.0] * dims
    for token in _tokens(text):
        vector[_bucket(token, dims)] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(x * y for x, y in zip(left, right, strict=True))


def _keyword_score(left: str, right: str) -> float:
    left_tokens, right_tokens = Counter(_tokens(left)), Counter(_tokens(right))
    overlap = sum(min(left_tokens[token], right_tokens[token]) for token in left_tokens)
    denominator = max(1, min(sum(left_tokens.values()), sum(right_tokens.values())))
    return overlap / denominator


def _entities(directive: ParsedDirective) -> set[str]:
    attrs = directive.attrs
    values = {
        attrs.get("subject", ""), attrs.get("character", ""),
        attrs.get("owner", ""), attrs.get("user", ""), attrs.get("actor", ""),
    }
    values.update(re.split(r"[,，]", attrs.get("participants", "")))
    return {value.strip() for value in values if value.strip()}


def _directive_text(directive: ParsedDirective) -> str:
    attrs = " ".join(f"{key} {value}" for key, value in sorted(directive.attrs.items()))
    return f"{directive.kind} {attrs} {directive.evidence.text}"


@dataclass(frozen=True, slots=True)
class CandidatePair:
    left_index: int
    right_index: int
    keyword_score: float
    vector_score: float
    graph_score: float
    total_score: float
    selected_reason: str

    def to_trace(self, directives: list[ParsedDirective], **extra) -> dict:
        left, right = directives[self.left_index], directives[self.right_index]
        return {
            "left": {"kind": left.kind, "document_name": left.evidence.document_name, "line": left.evidence.line_start},
            "right": {"kind": right.kind, "document_name": right.evidence.document_name, "line": right.evidence.line_start},
            "keyword_score": round(self.keyword_score, 6),
            "vector_score": round(self.vector_score, 6),
            "graph_score": round(self.graph_score, 6),
            "total_score": round(self.total_score, 6),
            "selected_reason": self.selected_reason,
            **extra,
        }


class HybridRetriever:
    """Reproducible local keyword + stable-hash n-gram + entity-graph retrieval."""

    def rank(self, query: str, candidates: list[EvidenceSpan], limit: int = 8) -> list[EvidenceSpan]:
        query_vector = _hash_vector(query)
        scored = []
        for item in candidates:
            keyword = _keyword_score(query, item.text)
            semantic = _cosine(query_vector, _hash_vector(item.text))
            scored.append((0.6 * keyword + 0.4 * semantic, item))
        return [item for _, item in sorted(scored, key=lambda row: (-row[0], row[1].document_name, row[1].line_start))[:limit]]

    def candidate_pairs(self, directives: list[ParsedDirective], *, threshold: float = 0.24, limit: int = 300) -> list[CandidatePair]:
        texts = [_directive_text(row) for row in directives]
        vectors = [_hash_vector(text) for text in texts]
        entities = [_entities(row) for row in directives]
        pairs: list[CandidatePair] = []
        for left_index, left in enumerate(directives):
            for right_index in range(left_index + 1, len(directives)):
                right = directives[right_index]
                if left.evidence.document_id == right.evidence.document_id and left.evidence.line_start == right.evidence.line_start and left.evidence.line_end == right.evidence.line_end:
                    continue
                keyword = _keyword_score(texts[left_index], texts[right_index])
                vector = _cosine(vectors[left_index], vectors[right_index])
                shared = entities[left_index] & entities[right_index]
                graph = 1.0 if shared else 0.0
                total = 0.45 * keyword + 0.35 * vector + 0.20 * graph
                if total < threshold and not shared:
                    continue
                reasons = []
                if shared:
                    reasons.append("shared_canonical_entity")
                if keyword >= 0.25:
                    reasons.append("keyword_overlap")
                if vector >= 0.25:
                    reasons.append("stable_ngram_similarity")
                pairs.append(CandidatePair(left_index, right_index, keyword, vector, graph, total, "+".join(reasons) or "combined_threshold"))
        return sorted(pairs, key=lambda row: (-row.total_score, row.left_index, row.right_index))[:limit]
