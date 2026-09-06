from __future__ import annotations

from collections import deque
import json
import unittest

import httpx

from langgraph.graph.state import CompiledStateGraph

from app.config import Settings
from app.domain import (
    AnalysisCancelled,
    EvidenceSpan,
    ModelExecutionDiagnostics,
    ParsedDirective,
)
from app.model_extractor import ModelEnhancedExtractor
from app.pipeline import AnalysisPipeline, DocumentInput
from app.provider import ModelResult, OpenAICompatibleProvider, ProviderError
from app.review_agent import (
    AGENT_SYSTEM_PROMPT,
    AgentCandidate,
    AgentTraceEvent,
    BoundedReviewAgent,
    MAX_SAFE_AGENT_LINE_NUMBER,
    ProtocolParseDiagnostic,
)


def settings(**overrides) -> Settings:
    values = {
        "enable_model_extraction": True,
        "openai_api_key": "test-only",
        "provider_max_attempts": 1,
        "per_run_token_budget": 50_000,
        "review_agent_token_budget": 12_000,
        "review_agent_total_deadline_seconds": 30,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class ScriptedProvider:
    def __init__(self, responses, configured_settings=None, monotonic=None):
        self.settings = configured_settings or settings(enable_review_agent=True)
        self.responses = deque(responses)
        self.calls: list[tuple[str, str]] = []
        if monotonic is not None:
            self.monotonic = monotonic

    @property
    def configured(self):
        return True

    def fork_for_agent(self, bounded_settings):
        self.last_agent_settings = bounded_settings
        return self

    def fork_for_repair(self, bounded_settings):
        self.last_repair_settings = bounded_settings
        return self

    def complete(self, system, user):
        self.calls.append((system, user))
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        if callable(response):
            response = response(system, user)
        if isinstance(response, dict):
            response = json.dumps(response, ensure_ascii=False)
        return ModelResult(text=response, prompt_tokens=11, completion_tokens=7)


def candidate(*, evidence="林澈的身份是领航员。", raw=None, doc_ref="d1"):
    record = raw or {
        "kind": "fact",
        "subject": "林澈",
        "predicate": "身份",
        "value": "舰长",
        "source_line_start": 1,
        "source_line_end": 1,
        "modality": "asserted",
        "source_scope": "narrator",
        "certainty": "certain",
    }
    document = DocumentInput(
        id=f"document-{doc_ref}",
        name="story.md",
        content=f"{evidence}\n随后他进入驾驶舱。",
        role="chapter",
        scope="route-a",
    )
    return AgentCandidate(
        index=1,
        doc_ref=doc_ref,
        raw_hash="a" * 64,
        raw_record=record,
        error_codes=("lexical_support",),
        document=document,
        line_start=1,
        line_end=1,
        evidence_text=evidence,
    )


def accepting_validator(row: AgentCandidate, fields):
    if fields.get("value") != "领航员":
        raise ValueError("unsupported")
    return ParsedDirective(
        kind="fact",
        attrs={
            "subject": "林澈",
            "predicate": "身份",
            "value": "领航员",
            "modality": "asserted",
            "source_scope": "narrator",
            "certainty": "certain",
        },
        evidence=EvidenceSpan(
            document_id=row.document.id,
            document_name=row.document.name,
            line_start=row.line_start,
            line_end=row.line_end,
            text=row.evidence_text,
        ),
        provenance_sources=frozenset({"model"}),
    )


def patch_from_read(*, value="领航员", candidate_index=1, doc_ref="d1", fields=None):
    def response(_system, user):
        payload = json.loads(user)
        observations = [
            row
            for row in payload["tool_observations"]
            if row.get("tool") == "READ_SPAN"
            and row.get("candidate_index") == candidate_index
        ]
        span_id = observations[-1]["span_id"]
        return {
            "actions": [
                {
                    "action": "PATCH_RECORDS",
                    "patches": [
                        {
                            "candidate_index": candidate_index,
                            "doc_ref": doc_ref,
                            "span_id": span_id,
                            "fields": fields or {"value": value},
                        }
                    ],
                }
            ]
        }

    return response


def read_action(
    *, candidate_index=1, doc_ref="d1", line_start=1, line_end=1
):
    return {
        "action": "READ_SPAN",
        "requests": [
            {
                "candidate_index": candidate_index,
                "doc_ref": doc_ref,
                "line_start": line_start,
                "line_end": line_end,
            }
        ],
    }


class ReviewAgentUnitTests(unittest.TestCase):
    def run_agent(self, responses, **kwargs):
        provider = ScriptedProvider(
            responses, kwargs.pop("configured_settings", None)
        )
        agent = BoundedReviewAgent(
            provider=provider,
            candidates=kwargs.pop("candidates", [candidate()]),
            patch_validator=kwargs.pop("patch_validator", accepting_validator),
            remaining_run_tokens=kwargs.pop("remaining_run_tokens", 50_000),
            checkpoint=kwargs.pop("checkpoint", None),
            monotonic=kwargs.pop("monotonic", None),
        )
        self.assertIsInstance(agent.graph, CompiledStateGraph)
        return agent.run(), provider

    def test_model_dynamically_reads_then_patches_successfully(self):
        run, provider = self.run_agent(
            [
                {
                    "actions": [
                        read_action(line_end=2)
                    ]
                },
                patch_from_read(),
            ]
        )
        self.assertEqual([1], list(run.recovered))
        self.assertEqual((), run.unresolved_indexes)
        self.assertEqual(2, run.decision_rounds)
        self.assertEqual(2, run.tool_calls)
        self.assertEqual(1, run.span_read_count)
        self.assertEqual(
            ["DECISION", "READ_SPAN", "DECISION", "PATCH_RECORDS"],
            [row.action for row in run.trace],
        )
        self.assertRegex(run.trace[-1].span_hash or "", r"^[a-f0-9]{64}$")
        second_prompt = json.loads(provider.calls[1][1])
        first_prompt = json.loads(provider.calls[0][1])
        self.assertNotIn(
            "林澈的身份是领航员",
            json.dumps(first_prompt, ensure_ascii=False),
        )
        self.assertEqual("read_ok", second_prompt["tool_observations"][0]["result"])
        self.assertIn("林澈的身份是领航员", second_prompt["tool_observations"][0]["text"])

    def test_read_observation_marks_only_allowlisted_literal_fields(self):
        marker = "MALICIOUS_FIELD_MUST_NOT_BECOME_FEEDBACK"
        base = {
            "kind": "event",
            "source_line_start": 1,
            "source_line_end": 1,
            "modality": "asserted",
            "source_scope": "narrator",
            "certainty": "certain",
        }
        cases = (
            (
                "第三日，林澈与叶岚抵达钟楼。",
                {
                    **base,
                    "id": "evt-1",
                    "time": "第三日",
                    "location": "钟楼",
                    "participants": ["林澈", "叶岚"],
                },
                ["location", "participants", "time"],
            ),
            (
                f"林澈看见{marker}。",
                {
                    **base,
                    "id": "",
                    "time": 3,
                    "location": "",
                    "participants": ["林澈", 7],
                    "malicious_field": marker,
                },
                [],
            ),
            (
                "林澈抵达钟楼。",
                {
                    **base,
                    "id": "evt-2",
                    "time": None,
                    "location": "钟楼",
                    "participants": ["林澈", "苏晚"],
                },
                ["location"],
            ),
            (
                "林澈抵达钟楼。",
                {
                    **base,
                    "id": "evt-3",
                    "location": "钟楼",
                    "participants": ["林澈", ""],
                },
                ["location"],
            ),
            (
                " 林澈抵达。",
                {
                    **base,
                    "id": " ",
                    "time": "\t",
                    "location": " ",
                    "participants": ["林澈", "   "],
                },
                [],
            ),
        )
        for evidence, raw, expected in cases:
            observed = []

            def inspect_observation(_system, user):
                observation = json.loads(user)["tool_observations"][-1]
                observed.extend(observation["literal_fields_already_present"])
                return {
                    "actions": [
                        {
                            "action": "ABSTAIN",
                            "candidate_indexes": [1],
                            "reason_code": "insufficient_evidence",
                        }
                    ]
                }

            with self.subTest(expected=expected):
                run, _ = self.run_agent(
                    [{"actions": [read_action()]}, inspect_observation],
                    candidates=[candidate(evidence=evidence, raw=raw)],
                )
                self.assertEqual(expected, observed)
                persisted_trace = json.dumps(run.safe_dict(), ensure_ascii=False)
                self.assertNotIn(evidence, persisted_trace)
                self.assertNotIn(marker, persisted_trace)

    def test_candidate_prompt_discloses_exact_server_read_bounds(self):
        run, provider = self.run_agent(
            [
                {
                    "actions": [
                        {
                            "action": "ABSTAIN",
                            "candidate_indexes": [1],
                            "reason_code": "insufficient_evidence",
                        }
                    ]
                }
            ],
            configured_settings=settings(
                enable_review_agent=True,
                review_agent_context_radius_lines=0,
            ),
        )
        self.assertEqual("explicit_abstain", run.final_reason)
        payload = json.loads(provider.calls[0][1])
        disclosed = payload["candidates"][0]
        self.assertEqual(2, disclosed["document_line_count"])
        self.assertEqual(
            {"line_start": 1, "line_end": 1}, disclosed["read_window"]
        )
        self.assertEqual(12, payload["limits"]["max_read_lines"])

    def test_read_must_be_fully_contained_in_disclosed_window(self):
        run, provider = self.run_agent(
            [{"actions": [read_action(line_start=1, line_end=2)]}],
            configured_settings=settings(
                enable_review_agent=True,
                review_agent_context_radius_lines=0,
            ),
        )
        disclosed = json.loads(provider.calls[0][1])["candidates"][0]
        self.assertEqual(
            {"line_start": 1, "line_end": 1}, disclosed["read_window"]
        )
        self.assertEqual("evidence_range", run.final_reason)
        self.assertEqual(0, run.tool_calls)
        self.assertEqual(0, run.span_read_count)
        rejected_read = next(row for row in run.trace if row.action == "READ_SPAN")
        self.assertEqual("rejected", rejected_read.final)
        self.assertEqual("evidence_range", rejected_read.validator_reason)
        self.assertEqual((1, 2), (rejected_read.line_start, rejected_read.line_end))
        self.assertEqual(
            (1, 1),
            (rejected_read.allowed_line_start, rejected_read.allowed_line_end),
        )
        self.assertIsNone(rejected_read.span_hash)
        safe_trace = json.dumps(run.safe_dict()["trace"], ensure_ascii=False)
        self.assertNotIn("林澈的身份是领航员", safe_trace)
        persisted = ModelExecutionDiagnostics(
            review_agent_runs=[run.safe_dict()]
        ).safe_dict()
        persisted_read = next(
            row
            for row in persisted["review_agent_runs"][0]["trace"]
            if row["action"] == "READ_SPAN"
        )
        self.assertEqual(1, persisted_read["allowed_line_start"])
        self.assertEqual(1, persisted_read["allowed_line_end"])

    def test_prompt_requires_minimal_changed_fields_without_silent_noop_masking(self):
        self.assertIn("只能列出相对候选值确实发生变化的最小字段", AGENT_SYSTEM_PROMPT)
        self.assertIn("不得重复提交值未变化的字段", AGENT_SYSTEM_PROMPT)
        self.assertIn("limits.max_read_lines", AGENT_SYSTEM_PROMPT)
        self.assertIn("literal_fields_already_present", AGENT_SYSTEM_PROMPT)
        self.assertIn("未列出的字段不代表一定错误", AGENT_SYSTEM_PROMPT)

    def test_trace_line_numbers_are_bounded_at_product_and_persistence_layers(self):
        huge = MAX_SAFE_AGENT_LINE_NUMBER + 1
        event = AgentTraceEvent(
            action="READ_SPAN",
            round=1,
            line_start=huge,
            line_end=-1,
            allowed_line_start=True,  # type: ignore[arg-type]
            allowed_line_end=huge,
            validator_reason="evidence_range",
            final="rejected",
        ).safe_dict()
        self.assertIsNone(event["line_start"])
        self.assertIsNone(event["line_end"])
        self.assertIsNone(event["allowed_line_start"])
        self.assertIsNone(event["allowed_line_end"])

        persisted = ModelExecutionDiagnostics(
            review_agent_runs=[
                {
                    "trace": [
                        {
                            "action": "READ_SPAN",
                            "round": 1,
                            "line_start": huge,
                            "line_end": huge,
                            "allowed_line_start": huge,
                            "allowed_line_end": huge,
                            "validator_reason": "evidence_range",
                            "final": "rejected",
                        }
                    ]
                }
            ]
        ).safe_dict()["review_agent_runs"][0]["trace"][0]
        for field in (
            "line_start",
            "line_end",
            "allowed_line_start",
            "allowed_line_end",
        ):
            self.assertIsNone(persisted[field])

    def test_invalid_action_persists_only_allowlisted_parse_diagnostics(self):
        marker = "DO_NOT_PERSIST_STORY_OR_SECRET_KEY"
        run, _ = self.run_agent(
            [
                {
                    "actions": [
                        {
                            "action": "ABSTAIN",
                            "candidate_indexes": [1],
                            "reason_code": "unsafe_patch",
                            "untrusted_field_name": marker,
                        }
                    ]
                }
            ]
        )
        self.assertEqual("invalid_action", run.final_reason)
        event = run.safe_dict()["trace"][0]
        self.assertEqual("invalid_action", event["validator_reason"])
        self.assertEqual(
            {
                "stage": "action_schema",
                "root_shape": "object",
                "action_count": 1,
                "action_names": ["ABSTAIN"],
                "schema_error_locations": ["unknown_field"],
                "schema_error_types": ["extra_forbidden"],
            },
            event["protocol_diagnostic"],
        )
        self.assertNotIn(marker, json.dumps(run.safe_dict(), ensure_ascii=False))

    def test_invalid_envelope_diagnostic_does_not_relax_protocol(self):
        run, _ = self.run_agent(
            [{"actions": [read_action()], "explanation": "not allowed"}]
        )
        self.assertEqual("invalid_action", run.final_reason)
        diagnostic = run.safe_dict()["trace"][0]["protocol_diagnostic"]
        self.assertEqual("envelope", diagnostic["stage"])
        self.assertEqual("object", diagnostic["root_shape"])
        self.assertEqual(0, run.tool_calls)

    def test_protocol_diagnostic_object_rejects_nested_and_oversized_values(self):
        marker = "DO_NOT_PERSIST_NESTED_VALUE"
        oversized = (marker + ".") * 10_000
        diagnostic = ProtocolParseDiagnostic(
            stage=[marker],  # type: ignore[arg-type]
            root_shape={"nested": marker},  # type: ignore[arg-type]
            action_names=([marker], {"nested": marker}, "ABSTAIN"),  # type: ignore[arg-type]
            schema_error_locations=(oversized, {"nested": marker}),  # type: ignore[arg-type]
            schema_error_types=([marker], {"nested": marker}, "missing"),  # type: ignore[arg-type]
        )
        safe = diagnostic.safe_dict()
        serialized = json.dumps(safe, ensure_ascii=False)
        self.assertNotIn(marker, serialized)
        self.assertLess(len(serialized), 1_000)
        self.assertEqual("action_schema", safe["stage"])
        self.assertEqual("unparsed", safe["root_shape"])
        self.assertEqual(
            ["unknown", "unknown", "ABSTAIN"], safe["action_names"]
        )
        self.assertEqual(
            ["validation_error", "validation_error", "missing"],
            safe["schema_error_types"],
        )

    def test_protocol_diagnostic_rejects_non_sequence_containers(self):
        marker = "DO_NOT_PERSIST_CONTAINER_VALUE"
        for container in (
            {"nested": marker},
            {marker},
            marker,
            123,
        ):
            with self.subTest(container_type=type(container).__name__):
                diagnostic = ProtocolParseDiagnostic(
                    stage="action_schema",
                    root_shape="object",
                    action_names=container,  # type: ignore[arg-type]
                    schema_error_locations=container,  # type: ignore[arg-type]
                    schema_error_types=container,  # type: ignore[arg-type]
                ).safe_dict()
                self.assertEqual([], diagnostic["action_names"])
                self.assertEqual([], diagnostic["schema_error_locations"])
                self.assertEqual([], diagnostic["schema_error_types"])
                self.assertNotIn(
                    marker, json.dumps(diagnostic, ensure_ascii=False)
                )

    def test_non_string_action_names_fail_closed_without_leaking_values(self):
        marker = "DO_NOT_PERSIST_UNHASHABLE_ACTION_VALUE"
        for action_value in ([marker], {"nested": marker}):
            with self.subTest(action_type=type(action_value).__name__):
                run, _ = self.run_agent(
                    [{"actions": [{"action": action_value}]}]
                )
                self.assertEqual("unknown_tool", run.final_reason)
                self.assertEqual(0, run.tool_calls)
                diagnostic = run.safe_dict()["trace"][0]["protocol_diagnostic"]
                self.assertEqual("action_name", diagnostic["stage"])
                self.assertEqual(["unknown"], diagnostic["action_names"])
                self.assertNotIn(
                    marker, json.dumps(run.safe_dict(), ensure_ascii=False)
                )

    def test_model_can_abstain_without_a_read(self):
        run, _ = self.run_agent(
            [
                {
                    "actions": [
                        {
                            "action": "ABSTAIN",
                            "candidate_indexes": [1],
                            "reason_code": "insufficient_evidence",
                        }
                    ]
                }
            ]
        )
        self.assertEqual({}, run.recovered)
        self.assertEqual((1,), run.unresolved_indexes)
        self.assertEqual((1,), run.abstained_indexes)
        self.assertEqual("explicit_abstain", run.final_reason)
        self.assertEqual(1, run.decision_rounds)
        self.assertEqual("ABSTAIN", run.trace[-1].action)

    def test_read_then_failed_patch_forces_bounded_abstention(self):
        run, _ = self.run_agent(
            [
                {
                    "actions": [
                        read_action(),
                    ]
                },
                patch_from_read(value="无原文依据"),
            ]
        )
        self.assertEqual({}, run.recovered)
        self.assertEqual((1,), run.unresolved_indexes)
        self.assertEqual((1,), run.abstained_indexes)
        self.assertEqual("round_limit", run.final_reason)
        self.assertEqual("patch_validation_failed", run.trace[-2].validator_reason)
        self.assertEqual("FINALIZE", run.trace[-1].action)

    def test_unknown_tool_abstains_without_execution(self):
        run, _ = self.run_agent(
            [{"actions": [{"action": "SEARCH_WEB", "query": "secret"}]}]
        )
        self.assertEqual("unknown_tool", run.final_reason)
        self.assertEqual(0, run.tool_calls)
        self.assertEqual((1,), run.abstained_indexes)

    def test_cross_document_read_abstains_before_tool_execution(self):
        run, _ = self.run_agent(
            [
                {
                    "actions": [
                        read_action(doc_ref="d2")
                    ]
                }
            ]
        )
        self.assertEqual("cross_document", run.final_reason)
        self.assertEqual(0, run.tool_calls)
        self.assertEqual((1,), run.abstained_indexes)

    def test_span_ids_cannot_be_forged_reused_or_cross_candidates(self):
        forged_patch = {
            "actions": [
                {
                    "action": "PATCH_RECORDS",
                    "patches": [
                        {
                            "candidate_index": 1,
                            "doc_ref": "d1",
                            "span_id": "f" * 32,
                            "fields": {"value": "领航员"},
                        }
                    ],
                }
            ]
        }
        run, _ = self.run_agent([forged_patch])
        self.assertEqual("read_required", run.final_reason)
        self.assertEqual({}, run.recovered)

        captured: dict[str, str] = {}

        def capture_and_patch(_system, user):
            span_id = json.loads(user)["tool_observations"][-1]["span_id"]
            captured["span_id"] = span_id
            return {
                "actions": [
                    {
                        "action": "PATCH_RECORDS",
                        "patches": [
                            {
                                "candidate_index": 1,
                                "doc_ref": "d1",
                                "span_id": span_id,
                                "fields": {"value": "领航员"},
                            }
                        ],
                    }
                ]
            }

        successful, _ = self.run_agent(
            [
                {
                    "actions": [
                        read_action()
                    ]
                },
                capture_and_patch,
            ]
        )
        self.assertEqual([1], list(successful.recovered))

        replay = {
            "actions": [
                {
                    "action": "PATCH_RECORDS",
                    "patches": [
                        {
                            "candidate_index": 1,
                            "doc_ref": "d1",
                            "span_id": captured["span_id"],
                            "fields": {"value": "领航员"},
                        }
                    ],
                }
            ]
        }
        replayed, _ = self.run_agent([replay])
        self.assertEqual("read_required", replayed.final_reason)

        second = candidate(doc_ref="d2")
        second = AgentCandidate(
            index=2,
            doc_ref="d2",
            raw_hash="b" * 64,
            raw_record=dict(second.raw_record),
            error_codes=second.error_codes,
            document=second.document,
            line_start=second.line_start,
            line_end=second.line_end,
            evidence_text=second.evidence_text,
        )

        def cross_candidate_patch(_system, user):
            span_id = json.loads(user)["tool_observations"][-1]["span_id"]
            return {
                "actions": [
                    {
                        "action": "PATCH_RECORDS",
                        "patches": [
                            {
                                "candidate_index": 2,
                                "doc_ref": "d2",
                                "span_id": span_id,
                                "fields": {"value": "领航员"},
                            }
                        ],
                    }
                ]
            }

        crossed, _ = self.run_agent(
            [
                {
                    "actions": [
                        read_action()
                    ]
                },
                cross_candidate_patch,
            ],
            candidates=[candidate(), second],
        )
        self.assertEqual("invalid_span", crossed.final_reason)
        self.assertEqual({}, crossed.recovered)

    def test_repeated_tool_action_stops_the_loop(self):
        read = {"actions": [read_action()]}
        run, _ = self.run_agent([read, read])
        self.assertEqual("repeated_loop", run.final_reason)
        self.assertEqual(1, run.tool_calls)
        self.assertEqual((1,), run.abstained_indexes)

    def test_tool_and_span_budgets_fail_closed(self):
        two_reads = {
            "actions": [
                read_action(),
                {
                    "action": "ABSTAIN",
                    "candidate_indexes": [1],
                    "reason_code": "insufficient_evidence",
                },
            ]
        }
        run, _ = self.run_agent(
            [two_reads],
            configured_settings=settings(
                enable_review_agent=True, review_agent_max_tool_calls=1
            ),
        )
        self.assertEqual("tool_budget", run.final_reason)
        self.assertEqual(0, run.tool_calls)

        run, _ = self.run_agent(
            [
                {
                    "actions": [
                        read_action()
                    ]
                }
            ],
            configured_settings=settings(
                enable_review_agent=True, review_agent_max_span_chars=1
            ),
        )
        self.assertEqual("span_budget", run.final_reason)
        self.assertEqual(0, run.tool_calls)

    def test_batch_read_and_patch_counts_one_tool_each(self):
        second = candidate(doc_ref="d2")
        second = AgentCandidate(
            index=2,
            doc_ref="d2",
            raw_hash="b" * 64,
            raw_record=dict(second.raw_record),
            error_codes=second.error_codes,
            document=second.document,
            line_start=second.line_start,
            line_end=second.line_end,
            evidence_text=second.evidence_text,
        )

        def batch_patch(_system, user):
            spans = {
                row["candidate_index"]: row["span_id"]
                for row in json.loads(user)["tool_observations"]
            }
            return {
                "actions": [
                    {
                        "action": "PATCH_RECORDS",
                        "patches": [
                            {
                                "candidate_index": index,
                                "doc_ref": f"d{index}",
                                "span_id": spans[index],
                                "fields": {"value": "领航员"},
                            }
                            for index in (1, 2)
                        ],
                    }
                ]
            }

        run, _ = self.run_agent(
            [
                {
                    "actions": [
                        {
                            "action": "READ_SPAN",
                            "requests": [
                                {
                                    "candidate_index": index,
                                    "doc_ref": f"d{index}",
                                    "line_start": 1,
                                    "line_end": 1,
                                }
                                for index in (1, 2)
                            ],
                        }
                    ]
                },
                batch_patch,
            ],
            candidates=[candidate(), second],
        )
        self.assertEqual({1, 2}, set(run.recovered))
        self.assertEqual(2, run.tool_calls)
        self.assertEqual(2, run.span_read_count)

    def test_batch_read_over_count_and_cross_document_fail_atomically(self):
        second = candidate(doc_ref="d2")
        second = AgentCandidate(
            index=2,
            doc_ref="d2",
            raw_hash="b" * 64,
            raw_record=dict(second.raw_record),
            error_codes=second.error_codes,
            document=second.document,
            line_start=second.line_start,
            line_end=second.line_end,
            evidence_text=second.evidence_text,
        )
        requests = [
            {
                "candidate_index": index,
                "doc_ref": f"d{index}",
                "line_start": 1,
                "line_end": 1,
            }
            for index in (1, 2)
        ]
        run, _ = self.run_agent(
            [{"actions": [{"action": "READ_SPAN", "requests": requests}]}],
            candidates=[candidate(), second],
            configured_settings=settings(
                enable_review_agent=True,
                review_agent_max_read_requests_per_action=1,
            ),
        )
        self.assertEqual("span_count_budget", run.final_reason)
        self.assertEqual(0, run.tool_calls)
        self.assertEqual(0, run.span_read_count)

        cross_document = [dict(row) for row in requests]
        cross_document[1]["doc_ref"] = "d1"
        run, _ = self.run_agent(
            [
                {
                    "actions": [
                        {"action": "READ_SPAN", "requests": cross_document}
                    ]
                }
            ],
            candidates=[candidate(), second],
        )
        self.assertEqual("cross_document", run.final_reason)
        self.assertEqual(0, run.tool_calls)
        self.assertEqual(0, run.span_read_count)

    def test_token_budget_blocks_call_and_provider_error_abstains(self):
        run, provider = self.run_agent(
            [],
            configured_settings=settings(
                enable_review_agent=True, review_agent_token_budget=256
            ),
            remaining_run_tokens=256,
        )
        self.assertEqual("token_budget", run.final_reason)
        self.assertEqual([], provider.calls)

        error = ProviderError("do not persist", category="read_timeout")
        run, _ = self.run_agent([error])
        self.assertEqual("provider_error", run.final_reason)
        self.assertEqual((1,), run.abstained_indexes)

    def test_deadline_and_cancel_gates(self):
        clock_values = iter([0.0, 31.0, 31.0])
        run, provider = self.run_agent([], monotonic=lambda: next(clock_values))
        self.assertEqual("deadline", run.final_reason)
        self.assertEqual([], provider.calls)

        def cancelled():
            raise AnalysisCancelled("cancelled")

        with self.assertRaises(AnalysisCancelled):
            self.run_agent([], checkpoint=cancelled)

    def test_post_provider_cancellation_keeps_completed_call_accounted(self):
        checkpoint_calls = 0
        accounted: list[tuple[bool, int, int, int]] = []

        def cancel_after_provider():
            nonlocal checkpoint_calls
            checkpoint_calls += 1
            if checkpoint_calls == 2:
                raise AnalysisCancelled("cancelled after provider response")

        provider = ScriptedProvider(
            [
                {
                    "actions": [
                        {
                            "action": "ABSTAIN",
                            "candidate_indexes": [1],
                            "reason_code": "insufficient_evidence",
                        }
                    ]
                }
            ]
        )
        agent = BoundedReviewAgent(
            provider=provider,
            candidates=[candidate()],
            patch_validator=accepting_validator,
            provider_accounting=lambda _telemetry, succeeded, prompt, completion, charged: accounted.append(
                (succeeded, prompt, completion, charged)
            ),
            remaining_run_tokens=50_000,
            checkpoint=cancel_after_provider,
        )
        with self.assertRaises(AnalysisCancelled):
            agent.run()

        self.assertEqual(1, len(provider.calls))
        self.assertEqual(1, len(accounted))
        # The second checkpoint is immediately after accounting. Raising here
        # prevents the graph from entering the tool-execution node.
        self.assertEqual(2, checkpoint_calls)
        succeeded, prompt, completion, charged = accounted[0]
        self.assertTrue(succeeded)
        self.assertEqual((11, 7), (prompt, completion))
        self.assertGreaterEqual(charged, prompt + completion)

    def test_forbidden_patch_and_atomic_patch_failure_never_partially_commit(self):
        run, _ = self.run_agent(
            [
                {
                    "actions": [
                        read_action()
                    ]
                },
                patch_from_read(fields={"source_line_start": 2}),
            ]
        )
        self.assertEqual("patch_field_forbidden", run.final_reason)
        self.assertEqual({}, run.recovered)

        second = candidate(doc_ref="d2")
        second = AgentCandidate(
            index=2,
            doc_ref=second.doc_ref,
            raw_hash="b" * 64,
            raw_record=dict(second.raw_record),
            error_codes=second.error_codes,
            document=second.document,
            line_start=second.line_start,
            line_end=second.line_end,
            evidence_text=second.evidence_text,
        )
        run, _ = self.run_agent(
            [
                {
                    "actions": [
                        {
                            "action": "READ_SPAN",
                            "requests": [
                                {
                                    "candidate_index": 1,
                                    "doc_ref": "d1",
                                    "line_start": 1,
                                    "line_end": 1,
                                },
                                {
                                    "candidate_index": 2,
                                    "doc_ref": "d2",
                                    "line_start": 1,
                                    "line_end": 1,
                                },
                            ],
                        }
                    ]
                },
                lambda _system, user: {
                    "actions": [
                        {
                            "action": "PATCH_RECORDS",
                            "patches": [
                                {
                                    "candidate_index": 1,
                                    "doc_ref": "d1",
                                    "span_id": next(
                                        row["span_id"]
                                        for row in json.loads(user)["tool_observations"]
                                        if row.get("candidate_index") == 1
                                    ),
                                    "fields": {"value": "领航员"},
                                },
                                {
                                    "candidate_index": 2,
                                    "doc_ref": "d2",
                                    "span_id": next(
                                        row["span_id"]
                                        for row in json.loads(user)["tool_observations"]
                                        if row.get("candidate_index") == 2
                                    ),
                                    "fields": {"value": "无依据"},
                                },
                            ],
                        }
                    ]
                },
            ],
            candidates=[candidate(), second],
        )
        self.assertEqual({}, run.recovered)
        self.assertEqual((1, 2), run.abstained_indexes)

    def test_semantic_fields_are_explicitly_forbidden_by_the_server(self):
        run, provider = self.run_agent(
            [
                {"actions": [read_action()]},
                patch_from_read(
                    fields={"value": "领航员", "modality": "asserted"}
                ),
            ]
        )
        self.assertEqual("semantic_field_forbidden", run.final_reason)
        self.assertEqual({}, run.recovered)
        first_prompt = json.loads(provider.calls[0][1])
        allowlist = first_prompt["candidates"][0]["patch_field_allowlist"]
        self.assertTrue(
            {"modality", "source_scope", "certainty", "evidence_medium"}.isdisjoint(
                allowlist
            )
        )

    def test_persistable_trace_contains_hashes_not_sensitive_text(self):
        marker = "DO_NOT_PERSIST_STORY_OR_SECRET_KEY"
        oversized = (marker + ".") * 10_000
        run, _ = self.run_agent(
            [
                {
                    "actions": [
                        {
                            "action": "ABSTAIN",
                            "candidate_indexes": [1],
                            "reason_code": "unsafe_patch",
                        }
                    ]
                }
            ],
            candidates=[candidate(evidence=marker)],
        )
        serialized = json.dumps(run.safe_dict(), ensure_ascii=False)
        self.assertNotIn(marker, serialized)
        self.assertNotIn("test-only", serialized)
        self.assertNotIn("api.openai.com", serialized)
        self.assertIn("a" * 64, serialized)

        injected = ModelExecutionDiagnostics(
            review_agent_runs=[
                {
                    **run.safe_dict(),
                    "prompt": marker,
                    "endpoint": "https://secret.invalid",
                    "trace": [
                        {
                            **run.safe_dict()["trace"][0],
                            "raw_response": marker,
                            "patch_value": marker,
                            "protocol_diagnostic": {
                                "stage": [marker, {"nested": marker}],
                                "root_shape": {"nested": [marker]},
                                "action_count": 999,
                                "action_names": [
                                    [marker],
                                    {"nested": marker},
                                    "ABSTAIN",
                                ],
                                "schema_error_locations": [
                                    oversized,
                                    [marker],
                                    {"nested": marker},
                                ],
                                "schema_error_types": [
                                    [marker],
                                    {"nested": marker},
                                    "extra_forbidden",
                                ],
                            },
                        }
                    ],
                }
            ]
        ).safe_dict()
        self.assertNotIn(marker, json.dumps(injected, ensure_ascii=False))
        self.assertNotIn("secret.invalid", json.dumps(injected, ensure_ascii=False))
        diagnostic = injected["review_agent_runs"][0]["trace"][0][
            "protocol_diagnostic"
        ]
        self.assertEqual("action_schema", diagnostic["stage"])
        self.assertEqual("unparsed", diagnostic["root_shape"])
        self.assertEqual(7, diagnostic["action_count"])
        self.assertEqual(
            ["unknown", "unknown", "ABSTAIN"], diagnostic["action_names"]
        )
        self.assertEqual(
            [
                ".".join(["unknown_field"] * 6),
                "unknown_field",
                "unknown_field",
            ],
            diagnostic["schema_error_locations"],
        )
        self.assertEqual(
            ["validation_error", "validation_error", "extra_forbidden"],
            diagnostic["schema_error_types"],
        )
        self.assertLess(len(json.dumps(diagnostic, ensure_ascii=False)), 1_000)

        base_run = run.safe_dict()
        base_event = base_run["trace"][0]
        bounded = ModelExecutionDiagnostics(
            review_agent_runs=[
                {**base_run, "trace": [dict(base_event) for _ in range(65)]}
                for _ in range(9)
            ]
        ).safe_dict()
        self.assertEqual(9, bounded["review_agent_total_runs"])
        self.assertTrue(bounded["review_agent_runs_truncated"])
        self.assertEqual(8, len(bounded["review_agent_runs"]))
        self.assertEqual(65, bounded["review_agent_runs"][0]["total_trace_events"])
        self.assertTrue(bounded["review_agent_runs"][0]["trace_truncated"])
        self.assertEqual(64, len(bounded["review_agent_runs"][0]["trace"]))


class ReviewAgentIntegrationTests(unittest.TestCase):
    def test_pipeline_keeps_attempted_abstained_and_succeeded_distinct(self):
        provider = ScriptedProvider(
            [
                {
                    "records": [
                        {
                            "kind": "fact",
                            "subject": "林澈",
                            "predicate": "身份",
                            "value": "舰长",
                            "source_line_start": 1,
                            "source_line_end": 1,
                            "modality": "asserted",
                            "source_scope": "narrator",
                            "certainty": "certain",
                        }
                    ]
                },
                {
                    "actions": [
                        {
                            "action": "ABSTAIN",
                            "candidate_indexes": [1],
                            "reason_code": "insufficient_evidence",
                        }
                    ]
                },
            ],
            settings(enable_review_agent=True),
        )
        result = AnalysisPipeline(
            extractor=ModelEnhancedExtractor(provider=provider)
        ).run(
            [
                DocumentInput(
                    id="doc-1",
                    name="chapter.md",
                    content="林澈的身份是领航员。",
                    role="chapter",
                    scope="route-a",
                )
            ]
        )
        status = result.diagnostics["model"]
        self.assertTrue(status["review_agent_attempted"])
        self.assertFalse(status["review_agent_succeeded"])
        self.assertTrue(status["review_agent_abstained"])

    def test_real_provider_fallback_instruments_agent_calls_once_with_agent_purpose(self):
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            payload = json.loads(request.content)
            system = payload["messages"][0]["content"]
            user = payload["messages"][1]["content"]
            if "叙事状态抽取器" in system:
                content = json.dumps(
                    {
                        "records": [
                            {
                                "kind": "fact",
                                "subject": "林澈",
                                "predicate": "身份",
                                "value": "舰长",
                                "source_line_start": 1,
                                "source_line_end": 1,
                                "modality": "asserted",
                                "source_scope": "narrator",
                                "certainty": "certain",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            else:
                agent_input = json.loads(user)
                observations = agent_input["tool_observations"]
                if not observations:
                    content = json.dumps(
                        {"actions": [read_action()]}, ensure_ascii=False
                    )
                else:
                    content = json.dumps(
                        patch_from_read()("", user), ensure_ascii=False
                    )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": content}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 13, "completion_tokens": 8},
                },
            )

        configured = settings(
            enable_review_agent=True,
            openai_base_url="https://mock.invalid/v1",
        )
        provider = OpenAICompatibleProvider(
            configured, transport=httpx.MockTransport(handler)
        )
        extractor = ModelEnhancedExtractor(provider=provider)
        extractor.begin_run()
        parsed = extractor.extract(
            DocumentInput(
                id="doc-1",
                name="chapter.md",
                content="林澈的身份是领航员。",
                role="chapter",
                scope="route-a",
            )
        )
        calls = parsed.model_execution.safe_dict()["provider_calls"]
        self.assertEqual(["extract", "agent", "agent"], [row["purpose"] for row in calls])
        self.assertEqual(3, call_count)
        self.assertTrue(parsed.model_execution.review_agent_succeeded)
        self.assertFalse(parsed.model_execution.repair_attempted)
        safe_accounting = extractor.review_agent_safe_accounting()
        self.assertEqual(2, safe_accounting["logical_calls"])
        self.assertEqual(
            extractor._review_agent_charged_tokens_used,
            safe_accounting["charged_tokens"],
        )
        self.assertEqual(
            2,
            len(
                [
                    row
                    for row in safe_accounting["provider_calls"]
                    if row["purpose"] == "agent"
                ]
            ),
        )

    def test_agent_deadline_starts_lazily_then_is_shared_across_documents(self):
        class Clock:
            value = 0.0

            def __call__(self):
                return self.value

        clock = Clock()
        extraction = {
            "records": [
                {
                    "kind": "fact",
                    "subject": "林澈",
                    "predicate": "身份",
                    "value": "舰长",
                    "source_line_start": 1,
                    "source_line_end": 1,
                    "modality": "asserted",
                    "source_scope": "narrator",
                    "certainty": "certain",
                }
            ]
        }

        def slow_main_extraction(_system, _user):
            clock.value = 100.0
            return extraction

        provider = ScriptedProvider(
            [
                slow_main_extraction,
                {
                    "actions": [
                        {
                            "action": "ABSTAIN",
                            "candidate_indexes": [1],
                            "reason_code": "insufficient_evidence",
                        }
                    ]
                },
                extraction,
            ],
            settings(
                enable_review_agent=True,
                model_circuit_breaker_failed_documents=99,
            ),
            monotonic=clock,
        )
        extractor = ModelEnhancedExtractor(provider=provider)
        extractor.begin_run()
        self.assertIsNone(extractor._review_agent_deadline)
        first = extractor.extract(
            DocumentInput(
                id="doc-1",
                name="a.md",
                content="林澈的身份是领航员。",
                role="chapter",
                scope="route-a",
            )
        )
        self.assertTrue(first.model_execution.review_agent_attempted)
        self.assertEqual(130.0, extractor._review_agent_deadline)

        clock.value = 131.0
        second = extractor.extract(
            DocumentInput(
                id="doc-2",
                name="b.md",
                content="林澈的身份是领航员。",
                role="chapter",
                scope="route-b",
            )
        )
        self.assertEqual("deadline", second.model_execution.review_agent_runs[0]["final_reason"])
        # Slow extraction before the first Agent did not consume its deadline;
        # after it started, the same absolute deadline applied to document two.
        self.assertEqual(3, len(provider.calls))

    def test_extractor_accounts_agent_call_before_post_provider_cancellation(self):
        extraction = {
            "records": [
                {
                    "kind": "fact",
                    "subject": "林澈",
                    "predicate": "身份",
                    "value": "舰长",
                    "source_line_start": 1,
                    "source_line_end": 1,
                    "modality": "asserted",
                    "source_scope": "narrator",
                    "certainty": "certain",
                }
            ]
        }
        provider = ScriptedProvider(
            [
                extraction,
                {
                    "actions": [
                        {
                            "action": "ABSTAIN",
                            "candidate_indexes": [1],
                            "reason_code": "insufficient_evidence",
                        }
                    ]
                },
            ],
            settings(enable_review_agent=True),
        )

        def cancel_after_agent_provider_call():
            if len(provider.calls) >= 2:
                raise AnalysisCancelled("cancelled after Agent response")

        extractor = ModelEnhancedExtractor(provider=provider)
        extractor.begin_run(checkpoint=cancel_after_agent_provider_call)
        with self.assertRaises(AnalysisCancelled):
            extractor.extract(
                DocumentInput(
                    id="doc-1",
                    name="chapter.md",
                    content="林澈的身份是领航员。",
                    role="chapter",
                    scope="route-a",
                )
            )

        self.assertEqual(2, len(provider.calls))
        self.assertGreater(extractor._review_agent_charged_tokens_used, 0)
        self.assertGreaterEqual(
            extractor._run_tokens_used, extractor._review_agent_charged_tokens_used
        )
        accounting = extractor.review_agent_safe_accounting()
        self.assertEqual(1, accounting["logical_calls"])
        self.assertEqual((11, 7), (
            accounting["prompt_tokens"], accounting["completion_tokens"]
        ))
        self.assertEqual(
            extractor._review_agent_charged_tokens_used,
            accounting["charged_tokens"],
        )
        pipeline = AnalysisPipeline(extractor=extractor)
        self.assertEqual(accounting, pipeline.interrupted_model_usage())
        # This scripted provider has no telemetry object. Missing telemetry is
        # explicit, never filled with fake zero-valued calls.
        self.assertIsNone(accounting["provider_calls"])

    def test_batch_agent_can_patch_a_subset_and_abstain_on_its_sibling(self):
        extraction = {
            "documents": [
                {
                    "doc_ref": "d1",
                    "records": [
                        {
                            "kind": "fact",
                            "subject": "林澈",
                            "predicate": "身份",
                            "value": "舰长",
                            "source_line_start": 1,
                            "source_line_end": 1,
                            "modality": "asserted",
                            "source_scope": "narrator",
                            "certainty": "certain",
                        }
                    ],
                },
                {
                    "doc_ref": "d2",
                    "records": [
                        {
                            "kind": "fact",
                            "subject": "苏晚",
                            "predicate": "身份",
                            "value": "领航员",
                            "source_line_start": 1,
                            "source_line_end": 1,
                            "modality": "asserted",
                            "source_scope": "narrator",
                            "certainty": "certain",
                        }
                    ],
                },
            ]
        }

        def patch_first_abstain_second(_system, user):
            observations = json.loads(user)["tool_observations"]
            spans = {
                row["candidate_index"]: row["span_id"]
                for row in observations
                if row.get("tool") == "READ_SPAN"
            }
            return {
                "actions": [
                    {
                        "action": "PATCH_RECORDS",
                        "patches": [
                            {
                                "candidate_index": 1,
                                "doc_ref": "d1",
                                "span_id": spans[1],
                                "fields": {"value": "领航员"},
                            }
                        ],
                    },
                    {
                        "action": "ABSTAIN",
                        "candidate_indexes": [2],
                        "reason_code": "insufficient_evidence",
                    },
                ]
            }

        provider = ScriptedProvider(
            [
                extraction,
                {
                    "actions": [
                        {
                            "action": "READ_SPAN",
                            "requests": [
                                {
                                    "candidate_index": 1,
                                    "doc_ref": "d1",
                                    "line_start": 1,
                                    "line_end": 1,
                                },
                                {
                                    "candidate_index": 2,
                                    "doc_ref": "d2",
                                    "line_start": 1,
                                    "line_end": 1,
                                },
                            ],
                        },
                    ]
                },
                patch_first_abstain_second,
            ],
            settings(enable_review_agent=True),
        )
        extractor = ModelEnhancedExtractor(provider=provider)
        extractor.begin_run()
        parsed = extractor.extract_batch(
            [
                DocumentInput(
                    id="doc-1",
                    name="a.md",
                    content="林澈的身份是领航员。",
                    role="chapter",
                    scope="route-a",
                ),
                DocumentInput(
                    id="doc-2",
                    name="b.md",
                    content="苏晚的身份是舰长。",
                    role="chapter",
                    scope="route-b",
                ),
            ]
        )
        self.assertIsNotNone(parsed)
        first, second = parsed
        self.assertEqual(1, first.model_execution.recovered_invalid_records)
        self.assertEqual(0, first.model_execution.unresolved_invalid_records)
        self.assertEqual(0, second.model_execution.recovered_invalid_records)
        self.assertEqual(1, second.model_execution.unresolved_invalid_records)
        self.assertTrue(first.model_execution.review_agent_succeeded)
        self.assertTrue(second.model_execution.review_agent_abstained)
        self.assertEqual(3, len(provider.calls))

    def test_model_extractor_routes_lexical_candidate_through_agent(self):
        extraction = {
            "records": [
                {
                    "kind": "fact",
                    "subject": "林澈",
                    "predicate": "身份",
                    "value": "舰长",
                    "source_line_start": 1,
                    "source_line_end": 1,
                    "modality": "asserted",
                    "source_scope": "narrator",
                    "certainty": "certain",
                }
            ]
        }
        provider = ScriptedProvider(
            [
                extraction,
                {
                    "actions": [
                        read_action()
                    ]
                },
                patch_from_read(),
            ],
            settings(enable_review_agent=True),
        )
        extractor = ModelEnhancedExtractor(provider=provider)
        extractor.begin_run()
        parsed = extractor.extract(
            DocumentInput(
                id="doc-1",
                name="chapter.md",
                content="林澈的身份是领航员。",
                role="chapter",
                scope="route-a",
            )
        )
        recovered = [
            row
            for row in parsed.directives
            if row.kind == "fact" and row.attrs.get("value") == "领航员"
        ]
        self.assertTrue(recovered)
        status = parsed.model_execution.safe_dict()
        self.assertEqual(1, status["invalid_records"])
        self.assertEqual(1, status["recovered_invalid_records"])
        self.assertEqual(0, status["unresolved_invalid_records"])
        self.assertTrue(status["review_agent_attempted"])
        self.assertTrue(status["review_agent_succeeded"])
        self.assertFalse(status["repair_attempted"])
        self.assertEqual("application_json_tools_v1", status["review_agent_runs"][0]["protocol"])
        self.assertEqual(3, len(provider.calls))

    def test_agent_cannot_promote_an_ineligible_candidate_during_lexical_repair(self):
        provider = ScriptedProvider(
            [
                {
                    "records": [
                        {
                            "kind": "fact",
                            "subject": "苏晚",
                            "predicate": "职务",
                            "value": "舰长官",
                            "source_line_start": 1,
                            "source_line_end": 1,
                            "modality": "asserted",
                            "source_scope": "narrator",
                            "certainty": "certain",
                        }
                    ]
                },
                {"actions": [read_action()]},
                patch_from_read(
                    fields={
                        "subject": "林澈",
                        "predicate": "身份",
                        "value": "领航员",
                    }
                ),
            ],
            settings(enable_review_agent=True),
        )
        extractor = ModelEnhancedExtractor(provider=provider)
        extractor.begin_run()
        parsed = extractor.extract(
            DocumentInput(
                id="doc-semantic-promotion",
                name="chapter.md",
                content="也许苏晚是舰长。林澈的身份是领航员。",
                role="chapter",
                scope="route-a",
            )
        )

        status = parsed.model_execution.safe_dict()
        self.assertEqual(0, status["recovered_invalid_records"])
        self.assertEqual(1, status["unresolved_invalid_records"])
        self.assertTrue(status["review_agent_abstained"])
        patch_events = [
            row
            for row in status["review_agent_runs"][0]["trace"]
            if row["action"] == "PATCH_RECORDS"
        ]
        self.assertEqual("semantic_promotion", patch_events[-1]["validator_reason"])

    def test_label_only_candidate_stays_on_fixed_repair_pass(self):
        provider = ScriptedProvider(
            [
                {
                    "records": [
                        {
                            "kind": "fact",
                            "subject": "林澈",
                            "predicate": "身份",
                            "value": "领航员",
                            "source_line_start": 1,
                            "source_line_end": 1,
                        }
                    ]
                },
                {
                    "patches": [
                        {
                            "record_index": 1,
                            "modality": "asserted",
                            "source_scope": "narrator",
                            "certainty": "certain",
                        }
                    ]
                },
            ],
            settings(enable_review_agent=True),
        )
        extractor = ModelEnhancedExtractor(provider=provider)
        extractor.begin_run()
        parsed = extractor.extract(
            DocumentInput(
                id="doc-1",
                name="chapter.md",
                content="林澈的身份是领航员。",
                role="chapter",
                scope="route-a",
            )
        )
        status = parsed.model_execution.safe_dict()
        self.assertTrue(status["repair_attempted"])
        self.assertTrue(status["repair_succeeded"])
        self.assertFalse(status["review_agent_attempted"])
        self.assertEqual(1, status["recovered_invalid_records"])
        self.assertEqual(0, status["unresolved_invalid_records"])
        self.assertEqual(2, len(provider.calls))

    def test_label_repair_and_lexical_agent_share_accounting_without_relabeling(self):
        provider = ScriptedProvider(
            [
                {
                    "records": [
                        {
                            "kind": "fact",
                            "subject": "林澈",
                            "predicate": "身份",
                            "value": "领航员",
                            "source_line_start": 1,
                            "source_line_end": 1,
                        },
                        {
                            "kind": "fact",
                            "subject": "苏晚",
                            "predicate": "身份",
                            "value": "领航员",
                            "source_line_start": 2,
                            "source_line_end": 2,
                            "modality": "asserted",
                            "source_scope": "narrator",
                            "certainty": "certain",
                        },
                    ]
                },
                {
                    "patches": [
                        {
                            "record_index": 1,
                            "modality": "asserted",
                            "source_scope": "narrator",
                            "certainty": "certain",
                        }
                    ]
                },
                {
                    "actions": [
                        read_action(candidate_index=2, line_start=2, line_end=2)
                    ]
                },
                patch_from_read(
                    candidate_index=2,
                    fields={"value": "舰长"},
                ),
            ],
            settings(enable_review_agent=True),
        )
        extractor = ModelEnhancedExtractor(provider=provider)
        extractor.begin_run()
        parsed = extractor.extract(
            DocumentInput(
                id="doc-1",
                name="chapter.md",
                content="林澈的身份是领航员。\n苏晚的身份是舰长。",
                role="chapter",
                scope="route-a",
            )
        )
        status = parsed.model_execution.safe_dict()
        self.assertEqual(2, status["invalid_records"])
        self.assertEqual(2, status["recovered_invalid_records"])
        self.assertEqual(0, status["unresolved_invalid_records"])
        self.assertTrue(status["repair_succeeded"])
        self.assertTrue(status["review_agent_succeeded"])
        self.assertEqual(4, len(provider.calls))

    def test_feature_off_preserves_legacy_lexical_rejection_without_agent_call(self):
        provider = ScriptedProvider(
            [
                {
                    "records": [
                        {
                            "kind": "fact",
                            "subject": "林澈",
                            "predicate": "身份",
                            "value": "舰长",
                            "source_line_start": 1,
                            "source_line_end": 1,
                            "modality": "asserted",
                            "source_scope": "narrator",
                            "certainty": "certain",
                        }
                    ]
                }
            ],
            settings(enable_review_agent=False),
        )
        extractor = ModelEnhancedExtractor(provider=provider)
        extractor.begin_run()
        parsed = extractor.extract(
            DocumentInput(
                id="doc-1",
                name="chapter.md",
                content="林澈的身份是领航员。",
                role="chapter",
                scope="route-a",
            )
        )
        status = parsed.model_execution.safe_dict()
        self.assertFalse(status["review_agent_attempted"])
        self.assertEqual([], status["review_agent_runs"])
        self.assertEqual(1, len(provider.calls))

    def test_agent_limits_are_shared_across_documents_in_one_analysis_run(self):
        extraction = {
            "records": [
                {
                    "kind": "fact",
                    "subject": "林澈",
                    "predicate": "身份",
                    "value": "舰长",
                    "source_line_start": 1,
                    "source_line_end": 1,
                    "modality": "asserted",
                    "source_scope": "narrator",
                    "certainty": "certain",
                }
            ]
        }
        provider = ScriptedProvider(
            [
                extraction,
                {
                    "actions": [
                        read_action()
                    ]
                },
                patch_from_read(),
                extraction,
            ],
            settings(enable_review_agent=True),
        )
        extractor = ModelEnhancedExtractor(provider=provider)
        extractor.begin_run()
        first = extractor.extract(
            DocumentInput(
                id="doc-1",
                name="a.md",
                content="林澈的身份是领航员。",
                role="chapter",
                scope="route-a",
            )
        )
        second = extractor.extract(
            DocumentInput(
                id="doc-2",
                name="b.md",
                content="林澈的身份是领航员。",
                role="chapter",
                scope="route-b",
            )
        )
        first_run = first.model_execution.review_agent_runs[0]
        second_run = second.model_execution.review_agent_runs[0]
        self.assertEqual(2, first_run["decision_rounds"])
        self.assertEqual(0, second_run["decision_rounds"])
        self.assertEqual(2, first_run["tool_calls"] + second_run["tool_calls"])
        self.assertLessEqual(
            first_run["span_chars"] + second_run["span_chars"],
            provider.settings.review_agent_max_span_chars,
        )
        self.assertLessEqual(
            first_run["span_read_count"] + second_run["span_read_count"],
            provider.settings.review_agent_max_span_reads,
        )
        self.assertEqual("round_limit", second_run["final_reason"])
        self.assertEqual(1, second.model_execution.unresolved_invalid_records)
        # Two extraction calls plus exactly two Agent decisions; the second
        # document cannot reset and spend another Agent round.
        self.assertEqual(4, len(provider.calls))


if __name__ == "__main__":
    unittest.main()
