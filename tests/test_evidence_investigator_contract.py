import hashlib
import json
import warnings
from dataclasses import replace

import pytest
from pydantic import ValidationError

from app.domain import EvidenceSpan, IssueCategory, ParsedDirective
from app.evidence_chunks import EvidenceChunker, SnapshotDocumentKey
from app.evidence_authority import (
    EvidenceGrantAuthority,
    InvestigationScope,
    ScopedEvidenceDocument,
)
from app.evidence_investigator import (
    AbstainArgs,
    CandidateRecordSubmission,
    InvestigationSeed,
    InvestigatorRejected,
    ReadSpanArgs,
    SearchEvidenceArgs,
    SubmitVerdictArgs,
    build_investigation_seeds,
    parse_tool_arguments,
)
from app.evidence_investigator_state import (
    EvidenceInvestigatorSession,
    InvestigatorLimits,
    InvestigatorRunResult,
    InvestigatorTraceEvent,
)


def directive(kind, attrs, *, document_id="doc-1", line=1, text=None):
    semantic = {
        "modality": "reported" if kind == "claims_knows" else "asserted",
        "source_scope": (
            "character_dialogue" if kind == "claims_knows" else "narrator"
        ),
        "certainty": "certain",
    }
    return ParsedDirective(
        kind=kind,
        attrs={**attrs, **semantic},
        evidence=EvidenceSpan(
            document_id=document_id,
            document_name="chapter.md",
            line_start=line,
            line_end=line,
            text=text or f"证据{line}",
        ),
        provenance_sources=frozenset({"model"}),
    )


def five_directives():
    return [
        directive("fact", {"subject": "岚", "predicate": "发色", "value": "银色"}),
        directive(
            "event",
            {
                "id": "e1",
                "time": "2026-01-01 20:00",
                "location": "北塔",
                "participants": "岚",
            },
            line=2,
        ),
        directive(
            "claims_knows",
            {"character": "岚", "fact": "潮门口令", "time": "20:00"},
            line=3,
        ),
        directive("item", {"item": "星钥", "owner": "岚", "time": "19:00"}, line=4),
        directive(
            "world_rule",
            {"key": "scope_action:北塔:跃迁", "value": "disabled"},
            line=5,
        ),
    ]


def scope_for(content="证据1\n证据2\n证据3\n证据4\n证据5", *, run_id="run-a"):
    snapshot = SnapshotDocumentKey(
        project_id="project-a",
        document_id="doc-1",
        document_version=1,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    document = ScopedEvidenceDocument(snapshot=snapshot, content=content)
    scope = InvestigationScope.create(
        run_id=run_id,
        project_id="project-a",
        documents=(document,),
    )
    return scope, document


def chunk_for(document, *, project_id=None):
    snapshot = document.snapshot
    return EvidenceChunker(
        target_chars=80,
        min_chars=1,
        max_chars=120,
        overlap_chars=0,
    ).chunk(
        project_id=project_id or snapshot.project_id,
        document_id=snapshot.document_id,
        document_version=snapshot.document_version,
        content=document.content,
        content_sha256=snapshot.content_sha256,
    )[0]


def fact_seed(run_id="run-a"):
    return build_investigation_seeds(run_id, [five_directives()[0]], limit=8)[0]


def candidate(span_ref, *, kind="fact", start=1, end=1, fields=None):
    return CandidateRecordSubmission.model_validate(
        {
            "kind": kind,
            "span_ref": span_ref,
            "source_line_start": start,
            "source_line_end": end,
            "fields": fields
            or {"subject": "岚", "predicate": "发色", "value": "黑色"},
        }
    )


def begin_search(session, seed, query="岚的发色是什么"):
    session.begin_decision(seed.seed_ref, charged_tokens=20)
    return SearchEvidenceArgs(
        seed_ref=seed.seed_ref,
        query=query,
        entity_terms=["岚"],
    )


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def test_seed_builder_covers_five_rule_families_and_is_stable_and_balanced():
    rows = five_directives()
    first = build_investigation_seeds("run-a", rows, limit=8)
    second = build_investigation_seeds("run-a", list(reversed(rows)), limit=8)

    assert [row.family for row in first] == list(IssueCategory)
    assert [row.seed_ref for row in first] == [row.seed_ref for row in second]
    assert all(row.run_hash == hashlib.sha256(b"run-a").hexdigest() for row in first)
    assert all("岚" not in repr(row) and "星钥" not in repr(row) for row in first)
    assert {
        row.family: row.allowed_candidate_kinds for row in first
    } == {
        IssueCategory.fact_conflict: frozenset({"fact"}),
        IssueCategory.location_collision: frozenset({"event"}),
        IssueCategory.knowledge_without_acquisition: frozenset(
            {"knows", "claims_knows"}
        ),
        IssueCategory.item_ownership: frozenset({"item", "uses"}),
        IssueCategory.world_rule_conflict: frozenset(
            {"world_rule", "world_assert"}
        ),
    }

    # The round-robin ordering keeps facts from consuming a small global cap.
    extras = [
        directive("fact", {"subject": f"人物{i}", "predicate": "发色", "value": "黑"})
        for i in range(10)
    ]
    limited = build_investigation_seeds("run-a", [*extras, *rows], limit=5)
    assert {row.family for row in limited} == set(IssueCategory)


def test_seed_builder_excludes_noncanonical_and_non_rule_fact_anchors():
    uncertain = directive("fact", {"subject": "岚", "predicate": "发色", "value": "银"})
    uncertain.attrs["certainty"] = "possible"
    mobility = directive(
        "fact",
        {"subject": "岚", "predicate": "mobility_limit", "value": "forbidden"},
    )
    question = directive("open_question", {"question": "岚在哪里？"})
    imprecise_event = directive(
        "event",
        {"time": "20:00", "location": "北塔", "participants": "岚"},
    )

    assert (
        build_investigation_seeds(
            "run-a", [uncertain, mobility, question, imprecise_event]
        )
        == ()
    )
    with pytest.raises(ValueError, match="seed limit"):
        build_investigation_seeds("run-a", [], limit=True)


@pytest.mark.parametrize("field", ["anchor_hash", "join_key_hash", "seed_ref", "family"])
def test_seed_identity_forgery_is_rejected(field):
    seed = fact_seed()
    changes = {
        "anchor_hash": {"anchor_hash": "0" * 64},
        "join_key_hash": {"join_key_hash": "0" * 64},
        "seed_ref": {"seed_ref": f"seed_{'0' * 32}"},
        "family": {
            "family": IssueCategory.item_ownership,
            "allowed_candidate_kinds": frozenset({"item", "uses"}),
        },
    }
    with pytest.raises(ValueError):
        replace(seed, **changes[field])


def test_seed_join_keys_use_unambiguous_canonical_arrays():
    colliding_under_nul_join = [
        directive(
            "fact",
            {"subject": "a", "predicate": "b\0c", "value": "one"},
        ),
        directive(
            "fact",
            {"subject": "a\0b", "predicate": "c", "value": "two"},
        ),
    ]
    fact_seeds = build_investigation_seeds(
        "run-a", colliding_under_nul_join, limit=8
    )
    assert len(fact_seeds) == 2
    assert len({row.seed_ref for row in fact_seeds}) == 2
    assert len({row.join_key_hash for row in fact_seeds}) == 2

    seeds = {
        row.family: row
        for row in build_investigation_seeds("run-a", five_directives(), limit=5)
    }

    def join_hash(parts):
        encoded = json.dumps(
            parts,
            ensure_ascii=False,
            sort_keys=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    assert seeds[IssueCategory.location_collision].join_key_hash == join_hash(
        ["event", "岚", "2026-01-01 20:00"]
    )
    assert seeds[IssueCategory.knowledge_without_acquisition].join_key_hash == join_hash(
        ["knowledge", "岚", "潮门口令"]
    )


def test_session_copies_seed_and_requires_exact_anchor_source_binding():
    scope, _ = scope_for()
    seed = fact_seed()
    session = EvidenceInvestigatorSession(scope=scope, seeds=(seed,))
    seed.anchor.attrs["subject"] = "外部修改"
    assert session._seeds[seed.seed_ref].anchor.attrs["subject"] == "岚"

    stale = fact_seed()
    stale.anchor.attrs["subject"] = "构造前污染"
    with pytest.raises(ValueError, match="session seeds"):
        EvidenceInvestigatorSession(scope=scope, seeds=(stale,))

    for unbound in (
        directive(
            "fact",
            {"subject": "岚", "predicate": "发色", "value": "银色"},
            text="不在快照中的原文",
        ),
        directive(
            "fact",
            {"subject": "岚", "predicate": "发色", "value": "银色"},
            line=6,
        ),
        directive(
            "fact",
            {"subject": "岚", "predicate": "发色", "value": "银色"},
            document_id="doc-other",
        ),
    ):
        unbound_seed = build_investigation_seeds("run-a", [unbound])[0]
        with pytest.raises(InvestigatorRejected, match="snapshot_mismatch"):
            EvidenceInvestigatorSession(scope=scope, seeds=(unbound_seed,))


def test_four_tool_argument_contracts_are_strict_and_have_no_scope_fields():
    seed_ref = fact_seed().seed_ref
    valid = {
        "SEARCH_EVIDENCE": {
            "seed_ref": seed_ref,
            "query": "岚的发色",
            "entity_terms": ["岚"],
        },
        "READ_SPAN": {
            "seed_ref": seed_ref,
            "result_ref": f"result_{'A' * 16}",
            "line_start": 1,
            "line_end": 2,
        },
        "SUBMIT_VERDICT": {
            "seed_ref": seed_ref,
            "verdict": "candidate_conflict",
            "candidates": [
                {
                    "kind": "fact",
                    "span_ref": f"span_{'B' * 16}",
                    "source_line_start": 1,
                    "source_line_end": 1,
                    "fields": {"subject": "岚", "predicate": "发色", "value": "黑"},
                }
            ],
        },
        "ABSTAIN": {"seed_ref": seed_ref, "reason": "insufficient_evidence"},
    }
    assert all(parse_tool_arguments(name, args) for name, args in valid.items())

    for name, args in valid.items():
        forged = {**args, "project_id": "another-project"}
        with pytest.raises(InvestigatorRejected) as caught:
            parse_tool_arguments(name, forged)
        assert caught.value.reason_code == "invalid_tool_arguments"
        assert "another-project" not in str(caught.value)

    for bad_line in (True, "1", 1.0):
        forged = {**valid["READ_SPAN"], "line_start": bad_line}
        with pytest.raises(InvestigatorRejected, match="invalid_tool_arguments"):
            parse_tool_arguments("READ_SPAN", forged)
    with pytest.raises(InvestigatorRejected, match="unknown_tool"):
        parse_tool_arguments("DELETE_DOCUMENT", {})
    with pytest.raises(InvestigatorRejected, match="unknown_tool"):
        parse_tool_arguments([], {})  # type: ignore[arg-type]
    with pytest.raises(InvestigatorRejected, match="invalid_tool_arguments"):
        parse_tool_arguments("SEARCH_EVIDENCE", [valid["SEARCH_EVIDENCE"]])


def test_malformed_exact_pydantic_inputs_fail_at_each_public_boundary():
    scope, document = scope_for()
    seed = fact_seed()

    session = EvidenceInvestigatorSession(scope=scope, seeds=(seed,))
    session.begin_decision(seed.seed_ref, charged_tokens=0)
    malformed_search = SearchEvidenceArgs.model_construct(
        query="岚的发色",
        entity_terms=["岚"],
    )
    with pytest.raises(InvestigatorRejected, match="invalid_tool_arguments"):
        session.search(malformed_search, ())

    authority = EvidenceGrantAuthority(scope=scope, seeds=(seed,))
    result = authority.issue_search_grants(seed.seed_ref, (chunk_for(document),))[0]
    malformed_read = ReadSpanArgs.model_construct(
        seed_ref=seed.seed_ref,
        result_ref=result.result_ref,
        line_start=1,
    )
    with pytest.raises(InvestigatorRejected, match="invalid_tool_arguments"):
        authority.read_span(malformed_read, max_read_lines=5)

    span = authority.read_span(
        ReadSpanArgs(
            seed_ref=seed.seed_ref,
            result_ref=result.result_ref,
            line_start=1,
            line_end=5,
        ),
        max_read_lines=5,
    )
    malformed_candidate = candidate(span.span_ref, start=1, end=1)
    object.__delattr__(malformed_candidate, "span_ref")
    with pytest.raises(InvestigatorRejected, match="invalid_tool_arguments"):
        authority.authorize_candidate(seed.seed_ref, malformed_candidate)

    malformed_directive = ParsedDirective.model_construct(kind="fact")
    with pytest.raises(ValueError, match="directive is invalid"):
        build_investigation_seeds("run-a", (malformed_directive,))


def test_mutated_pydantic_values_never_leak_through_warning_channels(
    capsys, caplog
):
    private = "PRIVATE-NARRATIVE-SERIALIZATION-VALUE"
    scope, document = scope_for()
    seed = fact_seed()
    caught_errors = []

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        session = EvidenceInvestigatorSession(scope=scope, seeds=(seed,))
        session.begin_decision(seed.seed_ref, charged_tokens=0)
        malformed_search = SearchEvidenceArgs(
            seed_ref=seed.seed_ref,
            query="岚的发色",
            entity_terms=["岚"],
        )
        object.__setattr__(malformed_search, "query", [private])
        with pytest.raises(
            InvestigatorRejected, match="invalid_tool_arguments"
        ) as caught:
            session.search(malformed_search, ())
        caught_errors.append(caught.value)

        authority = EvidenceGrantAuthority(scope=scope, seeds=(seed,))
        result = authority.issue_search_grants(
            seed.seed_ref, (chunk_for(document),)
        )[0]
        malformed_read = ReadSpanArgs(
            seed_ref=seed.seed_ref,
            result_ref=result.result_ref,
            line_start=1,
            line_end=1,
        )
        object.__setattr__(malformed_read, "line_start", [private])
        with pytest.raises(
            InvestigatorRejected, match="invalid_tool_arguments"
        ) as caught:
            authority.read_span(malformed_read, max_read_lines=5)
        caught_errors.append(caught.value)

        span = authority.read_span(
            ReadSpanArgs(
                seed_ref=seed.seed_ref,
                result_ref=result.result_ref,
                line_start=1,
                line_end=5,
            ),
            max_read_lines=5,
        )
        malformed_candidate = candidate(span.span_ref, start=1, end=1)
        object.__setattr__(malformed_candidate, "fields", [private])
        with pytest.raises(
            InvestigatorRejected, match="invalid_tool_arguments"
        ) as caught:
            authority.authorize_candidate(seed.seed_ref, malformed_candidate)
        caught_errors.append(caught.value)

        malformed_directive = directive(
            "fact", {"subject": "岚", "predicate": "发色", "value": "银色"}
        )
        object.__setattr__(malformed_directive.evidence, "text", [private])
        with pytest.raises(ValueError, match="directive is invalid") as caught:
            build_investigation_seeds("run-a", (malformed_directive,))
        caught_errors.append(caught.value)

    captured = capsys.readouterr()
    observable = "".join(
        (
            captured.out,
            captured.err,
            caplog.text,
            *(str(row.message) for row in caught_warnings),
            *(str(error) for error in caught_errors),
        )
    )
    assert private not in observable


@pytest.mark.parametrize("mutation", ["unhashable", "missing"])
def test_search_reconstructs_malformed_chunks_before_seen_state(mutation):
    scope, document = scope_for()
    seed = fact_seed()
    session = EvidenceInvestigatorSession(scope=scope, seeds=(seed,))
    chunk = chunk_for(document)
    if mutation == "unhashable":
        object.__setattr__(chunk, "chunk_id", [])
    else:
        object.__delattr__(chunk, "chunk_id")

    with pytest.raises(InvestigatorRejected, match="evidence_hash_mismatch"):
        session.search(begin_search(session, seed), (chunk,))
    assert session.result().failed_seed_reasons == {
        seed.seed_ref: "evidence_hash_mismatch"
    }


def test_candidate_transport_rejects_reserved_non_json_and_duplicate_payloads():
    seed_ref = fact_seed().seed_ref
    base = {
        "kind": "fact",
        "span_ref": f"span_{'B' * 16}",
        "source_line_start": 1,
        "source_line_end": 1,
        "fields": {"subject": "岚", "predicate": "发色", "value": "黑"},
    }
    for fields in (
        {"project_id": "forged"},
        {"title": "模型自拟问题"},
        {"nested": {"document_id": "forged"}},
        {"value": float("nan")},
        {"value": object()},
    ):
        with pytest.raises(ValidationError):
            CandidateRecordSubmission.model_validate({**base, "fields": fields})

    row = CandidateRecordSubmission.model_validate(base)
    with pytest.raises(ValidationError):
        SubmitVerdictArgs.model_validate(
            {
                "seed_ref": seed_ref,
                "verdict": "candidate_conflict",
                "candidates": [row.model_dump(), row.model_dump()],
            }
        )
    assert "模型自拟问题" not in repr(row)


def test_scope_rejects_hash_project_and_multiple_version_confusion():
    scope, document = scope_for()
    assert scope.project_id == "project-a"
    with pytest.raises(ValueError, match="hash mismatch"):
        ScopedEvidenceDocument(
            snapshot=document.snapshot,
            content=f"{document.content}tampered",
        )
    other = SnapshotDocumentKey(
        project_id="project-b",
        document_id="doc-b",
        document_version=1,
        content_sha256=hashlib.sha256(b"x").hexdigest(),
    )
    with pytest.raises(ValueError, match="another project"):
        InvestigationScope.create(
            run_id="run-a",
            project_id="project-a",
            documents=(ScopedEvidenceDocument(other, "x"),),
        )
    newer = SnapshotDocumentKey(
        project_id="project-a",
        document_id="doc-1",
        document_version=2,
        content_sha256=document.snapshot.content_sha256,
    )
    with pytest.raises(ValueError, match="duplicate document versions"):
        InvestigationScope.create(
            run_id="run-a",
            project_id="project-a",
            documents=(document, ScopedEvidenceDocument(newer, document.content)),
        )


def test_grant_authority_batch_preflight_is_atomic_and_snapshot_exact():
    scope, document = scope_for()
    seed = fact_seed()
    calls = 0

    def token():
        nonlocal calls
        calls += 1
        return f"token{calls:011d}ABCDE"

    authority = EvidenceGrantAuthority(
        scope=scope,
        seeds=(seed,),
        token_factory=token,
    )
    valid = chunk_for(document)
    other_content = document.content
    other_snapshot = SnapshotDocumentKey(
        project_id="project-b",
        document_id="doc-1",
        document_version=1,
        content_sha256=hashlib.sha256(other_content.encode()).hexdigest(),
    )
    other_document = ScopedEvidenceDocument(other_snapshot, other_content)
    foreign = chunk_for(other_document)

    with pytest.raises(InvestigatorRejected) as duplicate:
        authority.issue_search_grants(seed.seed_ref, (valid, valid))
    assert duplicate.value.reason_code == "snapshot_mismatch"
    with pytest.raises(InvestigatorRejected, match="snapshot_mismatch"):
        authority.issue_search_grants(seed.seed_ref, (valid, foreign))
    assert calls == 0
    assert authority.issue_search_grants(seed.seed_ref, (valid,))[0].result_ref.startswith(
        "result_"
    )

    wrong_lines = chunk_for(document)
    object.__setattr__(wrong_lines, "line_start", wrong_lines.line_start + 1)
    object.__setattr__(wrong_lines, "line_end", wrong_lines.line_end + 1)
    with pytest.raises(InvestigatorRejected, match="evidence_hash_mismatch"):
        authority.issue_search_grants(seed.seed_ref, (wrong_lines,))


def test_authority_read_is_atomic_and_enforces_the_session_line_limit():
    scope, document = scope_for()
    seed = fact_seed()
    authority = EvidenceGrantAuthority(scope=scope, seeds=(seed,))
    result = authority.issue_search_grants(seed.seed_ref, (chunk_for(document),))[0]
    args = ReadSpanArgs(
        seed_ref=seed.seed_ref,
        result_ref=result.result_ref,
        line_start=result.line_start,
        line_end=result.line_end,
    )

    assert not hasattr(authority, "prepare_span")
    assert not hasattr(authority, "issue_span")
    for invalid_limit in (True, 0, 21):
        with pytest.raises(InvestigatorRejected, match="evidence_range"):
            authority.read_span(args, max_read_lines=invalid_limit)
    with pytest.raises(InvestigatorRejected, match="evidence_range"):
        authority.read_span(args, max_read_lines=1)
    span = authority.read_span(args, max_read_lines=5)
    assert span.line_start == 1
    assert span.line_end == 5


def test_read_span_never_expands_a_partial_single_line_chunk_capability():
    authorized_text = "AUTHORIZED-" + "A" * 13
    private_neighbor = "PRIVATE-RIGHT-NEIGHBOR"
    assert len(authorized_text) == 24
    content = authorized_text + private_neighbor
    scope, document = scope_for(content)
    seed = build_investigation_seeds(
        "run-a",
        [
            directive(
                "fact",
                {"subject": "岚", "predicate": "发色", "value": "银色"},
                text=content,
            )
        ],
    )[0]
    chunks = EvidenceChunker(
        target_chars=20,
        min_chars=10,
        max_chars=24,
        overlap_chars=0,
    ).chunk(
        project_id=document.snapshot.project_id,
        document_id=document.snapshot.document_id,
        document_version=document.snapshot.document_version,
        content=document.content,
        content_sha256=document.snapshot.content_sha256,
    )
    assert len(chunks) >= 2
    assert chunks[0].line_start == chunks[0].line_end == 1

    session = EvidenceInvestigatorSession(scope=scope, seeds=(seed,))
    observation = session.search(begin_search(session, seed), (chunks[0],))[0]
    session.begin_decision(seed.seed_ref, charged_tokens=1)
    read = session.read(
        ReadSpanArgs(
            seed_ref=seed.seed_ref,
            result_ref=observation.result_ref,
            line_start=1,
            line_end=1,
        )
    )
    assert read.text == chunks[0].text == authorized_text
    assert private_neighbor not in read.text

    session.begin_decision(seed.seed_ref, charged_tokens=1)
    envelope = session.submit(
        SubmitVerdictArgs(
            seed_ref=seed.seed_ref,
            verdict="candidate_conflict",
            candidates=[candidate(read.span_ref, start=1, end=1)],
        )
    )
    assert envelope.candidates[0].source_line_start == 1
    assert envelope.candidates[0].source_line_end == 1


def test_session_detaches_scope_document_and_chunk_aliases():
    scope, document = scope_for()
    seed = fact_seed()
    chunk = chunk_for(document)
    original_text = chunk.text
    session = EvidenceInvestigatorSession(scope=scope, seeds=(seed,))

    object.__setattr__(document, "content", "EXTERNAL DOCUMENT MUTATION")
    object.__setattr__(scope, "project_id", "external-project")
    observation = session.search(begin_search(session, seed), (chunk,))[0]
    object.__setattr__(chunk, "text", "EXTERNAL CHUNK MUTATION")
    object.__setattr__(chunk.snapshot, "document_id", "external-document")

    session.begin_decision(seed.seed_ref, charged_tokens=1)
    read = session.read(
        ReadSpanArgs(
            seed_ref=seed.seed_ref,
            result_ref=observation.result_ref,
            line_start=observation.line_start,
            line_end=observation.line_end,
        )
    )
    assert read.text == original_text
    assert "EXTERNAL" not in read.text


def test_mutated_inputs_before_scope_or_chunk_boundary_fail_closed():
    scope, document = scope_for()
    seed = fact_seed()
    object.__setattr__(document, "content", "stale content")
    with pytest.raises(ValueError, match="scope is invalid"):
        EvidenceInvestigatorSession(scope=scope, seeds=(seed,))

    scope, document = scope_for()
    object.__setattr__(scope, "project_id", "another-project")
    with pytest.raises(ValueError, match="scope is invalid"):
        EvidenceInvestigatorSession(scope=scope, seeds=(seed,))

    scope, document = scope_for()
    session = EvidenceInvestigatorSession(scope=scope, seeds=(seed,))
    chunk = chunk_for(document)
    object.__setattr__(chunk, "text", "tampered before authorization")
    with pytest.raises(InvestigatorRejected, match="evidence_hash_mismatch"):
        session.search(begin_search(session, seed), (chunk,))


def test_authority_detaches_returned_search_and_span_grants():
    scope, document = scope_for()
    seed = fact_seed()
    authority = EvidenceGrantAuthority(scope=scope, seeds=(seed,))
    original_text = chunk_for(document).text
    result = authority.issue_search_grants(
        seed.seed_ref, (chunk_for(document),)
    )[0]
    result_ref = result.result_ref
    object.__setattr__(result, "char_start", result.char_end)
    object.__setattr__(result.snapshot, "document_id", "external-document")

    span = authority.read_span(
        ReadSpanArgs(
            seed_ref=seed.seed_ref,
            result_ref=result_ref,
            line_start=1,
            line_end=5,
        ),
        max_read_lines=5,
    )
    span_ref = span.span_ref
    object.__setattr__(span, "text", "EXTERNAL SPAN MUTATION")
    object.__setattr__(span.snapshot, "document_id", "external-span-document")
    authorized = authority.authorize_candidate(
        seed.seed_ref,
        candidate(span_ref, start=1, end=1),
    )
    assert authorized.text == original_text
    assert authorized.snapshot.document_id == "doc-1"


def test_fake_search_read_submit_path_only_returns_untrusted_envelope():
    secret_query = "PRIVATE-QUERY 岚的发色"
    scope, document = scope_for()
    seed = fact_seed()
    session = EvidenceInvestigatorSession(scope=scope, seeds=(seed,))

    search = begin_search(session, seed, secret_query)
    observations = session.search(search, (chunk_for(document),))
    search_action_hash = session._trace[-1].action_hash
    search.entity_terms.append("外部修改")
    assert session._trace[-1].action_hash == search_action_hash
    session.begin_decision(seed.seed_ref, charged_tokens=30)
    read = session.read(
        ReadSpanArgs(
            seed_ref=seed.seed_ref,
            result_ref=observations[0].result_ref,
            line_start=observations[0].line_start,
            line_end=observations[0].line_end,
        )
    )
    session.begin_decision(seed.seed_ref, charged_tokens=40)
    submitted = SubmitVerdictArgs(
        seed_ref=seed.seed_ref,
        verdict="candidate_conflict",
        candidates=[
            candidate(
                read.span_ref,
                start=read.line_start,
                end=read.line_start,
                fields={
                    "subject": "PRIVATE-CANDIDATE",
                    "predicate": "发色",
                    "value": "黑色",
                },
            )
        ],
    )
    envelope = session.submit(submitted)
    submitted.candidates[0].fields["value"] = "外部修改"
    submitted.candidates.clear()
    detached = envelope.candidates
    detached[0].fields["value"] = "再次修改"
    assert envelope.candidates[0].fields["value"] == "黑色"
    with pytest.raises(InvestigatorRejected, match="invalid_state"):
        session.begin_decision(seed.seed_ref, charged_tokens=1)
    assert session._terminal[seed.seed_ref] == "completed"
    result = session.result()

    assert envelope.trusted is False
    assert result.envelopes == (envelope,)
    assert not hasattr(envelope, "issues")
    assert result.safe_dict()["boundary"].startswith("SUBMIT_VERDICT returns untrusted")
    serialized = json.dumps(result.safe_dict(), ensure_ascii=False)
    rendered = f"{result!r}{envelope!r}{serialized}"
    assert secret_query not in rendered
    assert "PRIVATE-CANDIDATE" not in rendered
    assert document.content not in rendered
    assert result.safe_dict()["submitted_envelopes"] == 1
    assert result.safe_dict()["tool_calls"] == 3

    assert result.envelopes[0] is not envelope
    object.__setattr__(envelope, "candidate_payloads", ('{"leak":"submit"}',))
    object.__setattr__(
        result.envelopes[0], "candidate_payloads", ('{"leak":"result"}',)
    )
    object.__setattr__(result.trace[0], "reason_code", "PRIVATE TRACE")
    refetched = session.result()
    assert refetched.envelopes[0] is not result.envelopes[0]
    assert refetched.trace[0] is not result.trace[0]
    assert refetched.envelopes[0].candidates[0].fields["value"] == "黑色"
    assert refetched.trace[0].reason_code == "accepted"
    assert "PRIVATE TRACE" not in json.dumps(refetched.safe_dict(), ensure_ascii=False)


def test_abstain_is_terminal_and_separate_from_runtime_failure():
    scope, _ = scope_for()
    seed = fact_seed()
    session = EvidenceInvestigatorSession(scope=scope, seeds=(seed,))
    session.begin_decision(seed.seed_ref, charged_tokens=1)
    session.abstain(AbstainArgs(seed_ref=seed.seed_ref, reason="ambiguous_source"))
    with pytest.raises(InvestigatorRejected, match="invalid_state"):
        session.begin_decision(seed.seed_ref, charged_tokens=1)
    assert session._terminal[seed.seed_ref] == "explicit_abstain"
    result = session.result()
    assert result.abstained_seed_refs == (seed.seed_ref,)
    assert result.failed_seed_reasons == {}
    with pytest.raises(InvestigatorRejected, match="invalid_state"):
        session.begin_decision(seed.seed_ref, charged_tokens=1)


def test_wrong_claimed_seed_fails_only_the_current_pending_seed():
    scope, _ = scope_for()
    first, second = build_investigation_seeds("run-a", five_directives(), limit=2)
    session = EvidenceInvestigatorSession(scope=scope, seeds=(first, second))
    session.begin_decision(first.seed_ref, charged_tokens=1)

    with pytest.raises(InvestigatorRejected, match="invalid_state"):
        session.search(
            SearchEvidenceArgs(
                seed_ref=second.seed_ref,
                query="岚 发色",
                entity_terms=["岚"],
            ),
            (),
        )
    assert session._terminal == {first.seed_ref: "invalid_state"}
    with pytest.raises(InvestigatorRejected, match="invalid_state"):
        session.search(
            SearchEvidenceArgs(
                seed_ref=f"seed_{'f' * 32}",
                query="岚 发色",
                entity_terms=["岚"],
            ),
            (),
        )
    assert session._terminal == {first.seed_ref: "invalid_state"}
    result = session.result()
    assert result.failed_seed_reasons == {
        first.seed_ref: "invalid_state",
        second.seed_ref: "unprocessed_seed",
    }


def test_cross_run_cross_seed_forged_and_out_of_range_capabilities_fail_closed():
    scope, document = scope_for()
    seeds = build_investigation_seeds("run-a", five_directives(), limit=5)
    first, second = seeds[0], seeds[1]
    session = EvidenceInvestigatorSession(scope=scope, seeds=(first, second))
    search = begin_search(session, first)
    observation = session.search(search, (chunk_for(document),))[0]

    session.begin_decision(second.seed_ref, charged_tokens=1)
    with pytest.raises(InvestigatorRejected, match="cross_seed"):
        session.read(
            ReadSpanArgs(
                seed_ref=second.seed_ref,
                result_ref=observation.result_ref,
                line_start=observation.line_start,
                line_end=observation.line_end,
            )
        )

    clean = EvidenceInvestigatorSession(scope=scope, seeds=(first,))
    clean.begin_decision(first.seed_ref, charged_tokens=1)
    with pytest.raises(InvestigatorRejected, match="unknown_result_ref"):
        clean.read(
            ReadSpanArgs(
                seed_ref=first.seed_ref,
                result_ref=f"result_{'Z' * 16}",
                line_start=1,
                line_end=1,
            )
        )

    clean = EvidenceInvestigatorSession(scope=scope, seeds=(first,))
    observation = clean.search(begin_search(clean, first), (chunk_for(document),))[0]
    clean.begin_decision(first.seed_ref, charged_tokens=1)
    with pytest.raises(InvestigatorRejected, match="evidence_range"):
        clean.read(
            ReadSpanArgs(
                seed_ref=first.seed_ref,
                result_ref=observation.result_ref,
                line_start=observation.line_start,
                line_end=observation.line_end + 1,
            )
        )

    other_scope, _ = scope_for(run_id="run-b")
    with pytest.raises(InvestigatorRejected, match="cross_run"):
        EvidenceInvestigatorSession(scope=other_scope, seeds=(first,))


def test_submit_requires_real_same_seed_span_range_and_compatible_kind():
    scope, document = scope_for()
    seeds = build_investigation_seeds("run-a", five_directives(), limit=5)
    fact = next(row for row in seeds if row.family == IssueCategory.fact_conflict)
    location = next(
        row for row in seeds if row.family == IssueCategory.location_collision
    )

    session = EvidenceInvestigatorSession(scope=scope, seeds=(fact, location))
    observation = session.search(begin_search(session, fact), (chunk_for(document),))[0]
    session.begin_decision(fact.seed_ref, charged_tokens=1)
    read = session.read(
        ReadSpanArgs(
            seed_ref=fact.seed_ref,
            result_ref=observation.result_ref,
            line_start=observation.line_start,
            line_end=observation.line_end,
        )
    )
    session.begin_decision(fact.seed_ref, charged_tokens=1)
    with pytest.raises(InvestigatorRejected, match="candidate_kind_forbidden"):
        session.submit(
            SubmitVerdictArgs(
                seed_ref=fact.seed_ref,
                verdict="candidate_conflict",
                candidates=[
                    candidate(
                        read.span_ref,
                        kind="event",
                        start=read.line_start,
                        end=read.line_end,
                    )
                ],
            )
        )

    forged = EvidenceInvestigatorSession(scope=scope, seeds=(fact,))
    forged.begin_decision(fact.seed_ref, charged_tokens=1)
    with pytest.raises(InvestigatorRejected, match="unknown_span_ref"):
        forged.submit(
            SubmitVerdictArgs(
                seed_ref=fact.seed_ref,
                verdict="candidate_conflict",
                candidates=[candidate(f"span_{'X' * 16}")],
            )
        )


def test_repeat_query_and_no_progress_guards_are_run_state_not_prompt_advice():
    scope, document = scope_for()
    seed = fact_seed()
    session = EvidenceInvestigatorSession(scope=scope, seeds=(seed,))
    session.search(begin_search(session, seed, "ＡＢＣ  岚"), (chunk_for(document),))
    session.begin_decision(seed.seed_ref, charged_tokens=1)
    with pytest.raises(InvestigatorRejected, match="repeated_query"):
        session.search(
            SearchEvidenceArgs(
                seed_ref=seed.seed_ref,
                query="abc 岚",
                entity_terms=["岚"],
            ),
            (),
        )

    seed = fact_seed()
    session = EvidenceInvestigatorSession(scope=scope, seeds=(seed,))
    chunk = chunk_for(document)
    session.search(begin_search(session, seed, "岚 发色 一"), (chunk,))
    session.search(begin_search(session, seed, "岚 发色 二"), (chunk,))
    with pytest.raises(InvestigatorRejected, match="no_progress"):
        session.search(begin_search(session, seed, "岚 发色 三"), (chunk,))


@pytest.mark.parametrize(
    ("limits", "operation", "reason"),
    [
        (InvestigatorLimits(max_decision_rounds=1), "round", "round_budget"),
        (InvestigatorLimits(max_charged_tokens=1), "token", "token_budget"),
        (InvestigatorLimits(max_tool_calls=1), "tool", "tool_budget"),
        (InvestigatorLimits(max_searches=1), "search", "search_budget"),
        (InvestigatorLimits(max_results=1), "result", "result_budget"),
        (InvestigatorLimits(max_reads=1), "read", "read_budget"),
        (InvestigatorLimits(max_span_chars=1), "span", "span_budget"),
    ],
)
def test_all_run_global_budgets_fail_closed(limits, operation, reason):
    scope, document = scope_for()
    seed = fact_seed()
    chunk = chunk_for(document)
    session = EvidenceInvestigatorSession(scope=scope, seeds=(seed,), limits=limits)

    if operation in {"round", "token"}:
        session.begin_decision(seed.seed_ref, charged_tokens=1)
        session.search(
            SearchEvidenceArgs(
                seed_ref=seed.seed_ref,
                query="岚 发色 初次",
                entity_terms=["岚"],
            ),
            (),
        )
        with pytest.raises(InvestigatorRejected, match=reason):
            session.begin_decision(seed.seed_ref, charged_tokens=1)
        return

    if operation == "result":
        session.search(begin_search(session, seed), (chunk,))
        alternate = EvidenceChunker(
            target_chars=79,
            min_chars=1,
            max_chars=120,
            overlap_chars=0,
        ).chunk(
            project_id=document.snapshot.project_id,
            document_id=document.snapshot.document_id,
            document_version=document.snapshot.document_version,
            content=document.content,
            content_sha256=document.snapshot.content_sha256,
        )[0]
        session.begin_decision(seed.seed_ref, charged_tokens=0)
        with pytest.raises(InvestigatorRejected, match=reason):
            session.search(
                SearchEvidenceArgs(
                    seed_ref=seed.seed_ref,
                    query="岚 发色 多结果",
                    entity_terms=["岚"],
                ),
                (alternate,),
            )
        return

    observation = session.search(begin_search(session, seed), (chunk,))[0]
    if operation in {"tool", "search"}:
        session.begin_decision(seed.seed_ref, charged_tokens=0)
        with pytest.raises(InvestigatorRejected, match=reason):
            session.search(
                SearchEvidenceArgs(
                    seed_ref=seed.seed_ref,
                    query="岚 发色 再查",
                    entity_terms=["岚"],
                ),
                (),
            )
        return

    session.begin_decision(seed.seed_ref, charged_tokens=0)
    if operation == "span":
        with pytest.raises(InvestigatorRejected, match=reason):
            session.read(
                ReadSpanArgs(
                    seed_ref=seed.seed_ref,
                    result_ref=observation.result_ref,
                    line_start=observation.line_start,
                    line_end=observation.line_end,
                )
            )
        return
    first_read_end = (
        observation.line_start
        if operation == "read"
        else observation.line_end
    )
    session.read(
        ReadSpanArgs(
            seed_ref=seed.seed_ref,
            result_ref=observation.result_ref,
            line_start=observation.line_start,
            line_end=first_read_end,
        )
    )
    session.begin_decision(seed.seed_ref, charged_tokens=0)
    with pytest.raises(InvestigatorRejected, match=reason):
        session.read(
            ReadSpanArgs(
                seed_ref=seed.seed_ref,
                result_ref=observation.result_ref,
                line_start=observation.line_start + 1,
                line_end=observation.line_end,
            )
        )


def test_session_detaches_the_supplied_limits_object():
    scope, document = scope_for()
    seed = fact_seed()
    limits = InvestigatorLimits(max_searches=1)
    session = EvidenceInvestigatorSession(scope=scope, seeds=(seed,), limits=limits)
    object.__setattr__(limits, "max_searches", 100)

    session.search(begin_search(session, seed), (chunk_for(document),))
    session.begin_decision(seed.seed_ref, charged_tokens=0)
    with pytest.raises(InvestigatorRejected, match="search_budget"):
        session.search(
            SearchEvidenceArgs(
                seed_ref=seed.seed_ref,
                query="岚 发色 再查",
                entity_terms=["岚"],
            ),
            (),
        )


def test_missing_scope_and_limit_fields_fail_as_safe_constructor_errors():
    scope, _ = scope_for()
    seed = fact_seed()
    object.__delattr__(scope, "documents")
    with pytest.raises(ValueError, match="scope is invalid"):
        EvidenceInvestigatorSession(scope=scope, seeds=(seed,))

    scope, _ = scope_for()
    limits = InvestigatorLimits()
    object.__delattr__(limits, "max_searches")
    with pytest.raises(ValueError, match="limits are invalid"):
        EvidenceInvestigatorSession(scope=scope, seeds=(seed,), limits=limits)


def test_deadline_and_content_free_failure_trace_fail_closed():
    clock = Clock()
    scope, _ = scope_for()
    seed = fact_seed()
    session = EvidenceInvestigatorSession(
        scope=scope,
        seeds=(seed,),
        limits=InvestigatorLimits(deadline_seconds=1),
        monotonic=clock,
    )
    clock.value = 1.0
    with pytest.raises(InvestigatorRejected, match="deadline"):
        session.begin_decision(seed.seed_ref, charged_tokens=0)
    safe = json.dumps(session.result().safe_dict(), ensure_ascii=False)
    assert "deadline" in safe
    assert "run-a" not in safe
    assert "chapter.md" not in safe


def test_wrong_model_instance_is_rejected_without_execution():
    scope, _ = scope_for()
    seed = fact_seed()
    session = EvidenceInvestigatorSession(scope=scope, seeds=(seed,))
    session.begin_decision(seed.seed_ref, charged_tokens=0)
    with pytest.raises(InvestigatorRejected, match="invalid_tool_arguments"):
        session.read(  # type: ignore[arg-type]
            SearchEvidenceArgs(
                seed_ref=seed.seed_ref,
                query="岚 发色",
                entity_terms=["岚"],
            )
        )


def test_repeated_read_action_is_rejected_before_a_second_grant_is_issued():
    scope, document = scope_for()
    seed = fact_seed()
    session = EvidenceInvestigatorSession(scope=scope, seeds=(seed,))
    observation = session.search(begin_search(session, seed), (chunk_for(document),))[0]
    args = ReadSpanArgs(
        seed_ref=seed.seed_ref,
        result_ref=observation.result_ref,
        line_start=observation.line_start,
        line_end=observation.line_start,
    )
    session.begin_decision(seed.seed_ref, charged_tokens=0)
    session.read(args)
    session.begin_decision(seed.seed_ref, charged_tokens=0)
    with pytest.raises(InvestigatorRejected, match="repeated_action"):
        session.read(args)


def test_finalize_marks_unprocessed_seeds_and_is_irreversible():
    scope, _ = scope_for()
    first, second = build_investigation_seeds("run-a", five_directives(), limit=2)
    session = EvidenceInvestigatorSession(scope=scope, seeds=(first, second))
    session.begin_decision(first.seed_ref, charged_tokens=1)
    session.abstain(
        AbstainArgs(seed_ref=first.seed_ref, reason="no_relevant_evidence")
    )

    result = session.result()
    assert result.failed_seed_reasons == {second.seed_ref: "unprocessed_seed"}
    result.failed_seed_reasons[second.seed_ref] = "external mutation"
    assert session.result().failed_seed_reasons == {
        second.seed_ref: "unprocessed_seed"
    }
    with pytest.raises(InvestigatorRejected, match="invalid_state"):
        session.begin_decision(second.seed_ref, charged_tokens=1)
    assert session.result().safe_dict()["failure_reasons"] == ["unprocessed_seed"]


def test_safe_serializers_do_not_trust_manually_constructed_dataclasses():
    event = InvestigatorTraceEvent(
        action="unexpected",  # type: ignore[arg-type]
        seed_hash="plain seed",
        round_no=-1,
        outcome="unexpected",  # type: ignore[arg-type]
        reason_code="raw diagnostic detail",
        action_hash="plain action",
        result_count=-2,
        line_start=True,
        line_end=10_000_001,
        span_hash="plain span",
        charged_tokens="3",  # type: ignore[arg-type]
    )
    result = InvestigatorRunResult(
        envelopes="not a tuple",  # type: ignore[arg-type]
        completed_seed_refs=("not-a-seed",),
        abstained_seed_refs=(),
        failed_seed_reasons={
            f"seed_{'1' * 32}": "deadline",
            f"seed_{'2' * 32}": "raw failure detail",
            "not-a-seed": "token_budget",
        },
        decision_rounds=-1,
        tool_calls="2",  # type: ignore[arg-type]
        searches=-1,
        reads=-1,
        result_count=-1,
        span_chars=-1,
        charged_tokens=-1,
        trace=(event, "not-an-event"),  # type: ignore[arg-type]
    )

    safe = result.safe_dict()
    assert safe["decision_rounds"] == 0
    assert safe["tool_calls"] == 0
    assert safe["submitted_envelopes"] == 0
    assert safe["failed_seeds"] == 1
    assert safe["failure_reasons"] == ["deadline"]
    assert safe["trace"] == [
        {
            "action": "FINALIZE",
            "seed_hash": None,
            "round": 0,
            "outcome": "rejected",
            "reason_code": "internal_failure",
            "action_hash": None,
            "result_count": 0,
            "line_start": None,
            "line_end": None,
            "span_hash": None,
            "charged_tokens": 0,
        }
    ]
    assert "raw" not in json.dumps(safe, ensure_ascii=False)


def test_run_result_rejects_trace_subclass_dynamic_dispatch():
    class LeakingTraceEvent(InvestigatorTraceEvent):
        def safe_dict(self):
            return {"evidence": "PRIVATE NARRATIVE"}

    event = LeakingTraceEvent(
        action="DECISION",
        seed_hash="1" * 64,
        round_no=1,
        outcome="accepted",
        reason_code="accepted",
    )
    result = InvestigatorRunResult(
        envelopes=(),
        completed_seed_refs=(),
        abstained_seed_refs=(),
        failed_seed_reasons={},
        decision_rounds=1,
        tool_calls=0,
        searches=0,
        reads=0,
        result_count=0,
        span_chars=0,
        charged_tokens=0,
        trace=(event,),
    )

    safe = result.safe_dict()
    assert safe["trace"] == []
    assert safe["total_trace_events"] == 0
    assert "PRIVATE NARRATIVE" not in json.dumps(safe, ensure_ascii=False)
