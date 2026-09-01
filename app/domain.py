from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Severity(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class RunStatus(StrEnum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class IssueCategory(StrEnum):
    fact_conflict = "fact_conflict"
    location_collision = "location_collision"
    knowledge_without_acquisition = "knowledge_without_acquisition"
    item_ownership = "item_ownership"
    world_rule_conflict = "world_rule_conflict"


class EvidenceSpan(BaseModel):
    document_id: str
    document_name: str
    line_start: int
    line_end: int
    text: str


class NarrativeEntity(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    kind: str
    name: str
    aliases: list[str] = Field(default_factory=list)


class NarrativeEvent(BaseModel):
    id: str
    timestamp: str
    location: str
    participants: list[str]
    evidence: EvidenceSpan


class CanonicalFact(BaseModel):
    subject: str
    predicate: str
    value: str
    timestamp: str | None = None
    evidence: EvidenceSpan


class KnowledgeAcquisition(BaseModel):
    character: str
    fact: str
    timestamp: str
    evidence: EvidenceSpan


class ParsedDirective(BaseModel):
    kind: str
    attrs: dict[str, str]
    evidence: EvidenceSpan


class ConsistencyIssue(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    category: IssueCategory
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    title: str
    explanation: str
    evidence: list[EvidenceSpan]
    suggestion: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisRun(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    status: RunStatus = RunStatus.queued
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    input_chars: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0
    error: str | None = None


class EvaluationResult(BaseModel):
    sample_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    evidence_hit_rate: float
    category_scores: dict[str, dict[str, float | int]]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    issue_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str
    label: str
    record_id: str
    timestamp: str | None = None
    evidence: EvidenceSpan
    issue_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphResponse(BaseModel):
    run_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)


class TimelineEntry(BaseModel):
    id: str
    record_id: str
    kind: str
    title: str
    timestamp: str | None = None
    precision: str
    evidence: EvidenceSpan
    issue_ids: list[str] = Field(default_factory=list)
    attrs: dict[str, str] = Field(default_factory=dict)


class TimelineGroup(BaseModel):
    timestamp: str
    sort_key: str
    precision: str
    entries: list[TimelineEntry]


class TimelineResponse(BaseModel):
    run_id: str
    groups: list[TimelineGroup]
    unscheduled: list[TimelineEntry]
    warnings: list[str] = Field(default_factory=list)
