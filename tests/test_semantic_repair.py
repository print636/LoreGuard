import itertools
import json
import unittest

import httpx

from app.domain import AnalysisCancelled, EvidenceSpan, ParsedDirective
from app.model_extractor import ModelEnhancedExtractor
from app.parser import ParsedDocument
from app.pipeline import AnalysisPipeline, DocumentInput
from app.provider import OpenAICompatibleProvider, RetryPolicy
from app.semantic_quality import assess_directive, eligible_for_deterministic_rules
from app.service import analysis_mode
from tests.test_model_extractor import completion, settings


class EmptyBaseline:
    def extract(self, document):
        return ParsedDocument(document_id=document.id, document_name=document.name)


def fact(**updates):
    row = {
        "kind": "fact",
        "subject": "林澈",
        "predicate": "身份",
        "value": "领航员",
        "source_line_start": 1,
        "source_line_end": 1,
        "modality": "asserted",
        "source_scope": "narrator",
        "certainty": "certain",
    }
    row.update(updates)
    return row


def provider_for(responses, calls, **overrides):
    values = iter(responses)

    def handler(request):
        calls.append(json.loads(request.content))
        value = next(values)
        if isinstance(value, Exception):
            value.request = request
            raise value
        return completion(json.dumps(value, ensure_ascii=False))

    return OpenAICompatibleProvider(
        settings(provider_max_attempts=1, **overrides),
        transport=httpx.MockTransport(handler),
        retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0),
        sleep=lambda _: None,
    )


class SemanticRepairTests(unittest.TestCase):
    def run_single(self, raw, repair, text="林澈的身份是领航员。", **overrides):
        calls = []
        provider = provider_for(
            [{"records": [raw]}, repair], calls, **overrides
        )
        parsed = ModelEnhancedExtractor(provider, baseline=EmptyBaseline()).extract(
            DocumentInput("doc", "chapter.md", text)
        )
        return parsed, calls

    def test_every_invalid_label_combination_and_all_missing_are_repairable(self):
        labels = ("modality", "source_scope", "certainty")
        cases = []
        for size in range(1, 4):
            for fields in itertools.combinations(labels, size):
                raw = fact()
                for field in fields:
                    raw[field] = "illegal"
                cases.append((fields, raw))
        cases.append(("all_missing", {
            key: value for key, value in fact().items() if key not in labels
        }))
        patch = {"patches": [{
            "record_index": 1,
            "modality": "asserted",
            "source_scope": "narrator",
            "certainty": "certain",
        }]}
        for name, raw in cases:
            with self.subTest(case=name):
                parsed, calls = self.run_single(raw, patch)
                self.assertEqual(2, len(calls))
                self.assertTrue(parsed.model_execution.repair_succeeded)
                self.assertEqual("fact", parsed.directives[0].kind)
                self.assertEqual(
                    ["extract", "repair"],
                    [row.purpose for row in parsed.model_execution.provider_calls],
                )

    def test_probe_equivalent_four_missing_labels_are_raw_invalid_but_fully_recovered(self):
        identities = (("林澈", "领航员"), ("苏弦", "档案官"), ("顾青", "守卫"), ("陆遥", "调查员"))
        records = []
        patches = []
        lines = []
        for index, (subject, value) in enumerate(identities, start=1):
            records.append({
                "kind": "fact",
                "subject": subject,
                "predicate": "身份",
                "value": value,
                "source_line_start": index,
                "source_line_end": index,
            })
            patches.append({
                "record_index": index,
                "modality": "asserted",
                "source_scope": "narrator",
                "certainty": "certain",
            })
            lines.append(f"{subject}的身份是{value}。")
        calls = []
        provider = provider_for(
            [{"records": records}, {"patches": patches}], calls
        )
        result = AnalysisPipeline(
            extractor=ModelEnhancedExtractor(provider, baseline=EmptyBaseline())
        ).run([DocumentInput("probe", "probe.md", "\n".join(lines))])
        status = result.diagnostics["model"]
        self.assertEqual(2, len(calls))
        self.assertEqual((4, 0, 4), (
            status["invalid_records"],
            status["unresolved_invalid_records"],
            status["recovered_invalid_records"],
        ))
        self.assertEqual(0, status["repair_post_invalid"])
        self.assertFalse(status["repair_failed"])
        self.assertEqual("repaired", status["repair_final_path"])
        self.assertEqual(("完整模型增强", False), analysis_mode(result))

    def test_default_repair_payload_is_capability_neutral(self):
        raw = {key: value for key, value in fact().items() if key not in {
            "modality", "source_scope", "certainty"
        }}
        patch = {"patches": [{
            "record_index": 1,
            "modality": "asserted",
            "source_scope": "narrator",
            "certainty": "certain",
        }]}
        parsed, calls = self.run_single(raw, patch)
        self.assertTrue(parsed.model_execution.repair_succeeded)
        repair_payload = calls[1]
        self.assertTrue({
            "max_tokens",
            "max_completion_tokens",
            "reasoning",
            "reasoning_effort",
            "schema",
            "json_schema",
        }.isdisjoint(repair_payload))
        self.assertNotIn("provider_max_response_bytes", repair_payload)
        self.assertNotIn("schema", json.dumps(repair_payload["response_format"]))

    def test_repair_provider_enforces_64kb_network_cap_without_payload_change(self):
        calls = []
        provider = provider_for([], calls, provider_max_response_bytes=None)
        bounded = ModelEnhancedExtractor(
            provider, baseline=EmptyBaseline()
        )._bounded_repair_provider()
        self.assertEqual(64_000, bounded.settings.provider_max_response_bytes)

        provider = provider_for(
            [], calls,
            provider_max_response_bytes=48_000,
            semantic_repair_max_response_bytes=96_000,
        )
        bounded = ModelEnhancedExtractor(
            provider, baseline=EmptyBaseline()
        )._bounded_repair_provider()
        self.assertEqual(48_000, bounded.settings.provider_max_response_bytes)

    def test_repair_completion_cap_is_only_sent_when_explicit_and_uses_minimum(self):
        raw = {key: value for key, value in fact().items() if key not in {
            "modality", "source_scope", "certainty"
        }}
        patch = {"patches": [{
            "record_index": 1,
            "modality": "asserted",
            "source_scope": "narrator",
            "certainty": "certain",
        }]}
        _, calls = self.run_single(
            raw,
            patch,
            provider_max_completion_tokens=384,
            semantic_repair_max_completion_tokens=128,
        )
        self.assertEqual(384, calls[0]["max_tokens"])
        self.assertEqual(128, calls[1]["max_tokens"])

    def test_repair_response_bytes_remain_bounded_without_a_completion_cap(self):
        raw = {key: value for key, value in fact().items() if key not in {
            "modality", "source_scope", "certainty"
        }}
        parsed, calls = self.run_single(
            raw,
            {"patches": [{
                "record_index": 1,
                "modality": "asserted",
                "source_scope": "narrator",
                "certainty": "certain",
            }]},
            semantic_repair_max_response_bytes=32,
        )
        self.assertEqual(2, len(calls))
        self.assertTrue(parsed.model_execution.repair_failed)
        self.assertEqual([], parsed.directives)

    def test_baseline_provenance_alone_never_fills_any_missing_label(self):
        complete = {
            "modality": "asserted",
            "source_scope": "narrator",
            "certainty": "certain",
        }
        for size in range(1, 4):
            for fields in itertools.combinations(complete, size):
                attrs = {
                    "subject": "角色甲",
                    "predicate": "身份",
                    "value": "守卫",
                    **{key: value for key, value in complete.items() if key not in fields},
                }
                directive = ParsedDirective(
                    kind="fact",
                    attrs=attrs,
                    evidence=EvidenceSpan(
                        document_id="d",
                        document_name="d.md",
                        line_start=1,
                        line_end=1,
                        text="设定标签。",
                    ),
                    provenance_sources=frozenset({"baseline"}),
                )
                assessed, _ = assess_directive(directive)
                self.assertIsNotNone(assessed)
                self.assertFalse(eligible_for_deterministic_rules(assessed))

    def test_patch_must_exactly_cover_indexes_and_cannot_mutate_core(self):
        attacks = [
            {"patches": []},
            {"patches": [
                {"record_index": 1, "modality": "asserted", "source_scope": "narrator", "certainty": "certain"},
                {"record_index": 1, "modality": "asserted", "source_scope": "narrator", "certainty": "certain"},
            ]},
            {"patches": [{"record_index": 9, "modality": "asserted", "source_scope": "narrator", "certainty": "certain"}]},
            {"patches": [{"record_index": 1, "modality": "asserted", "source_scope": "narrator", "certainty": "certain", "kind": "world_rule"}]},
        ]
        raw = {key: value for key, value in fact().items() if key not in {
            "modality", "source_scope", "certainty"
        }}
        for attack in attacks:
            with self.subTest(attack=attack):
                parsed, calls = self.run_single(raw, attack)
                self.assertEqual(2, len(calls))
                self.assertTrue(parsed.model_execution.repair_failed)
                self.assertEqual(1, parsed.model_execution.repair_dropped)
                self.assertEqual(1, parsed.model_execution.invalid_records)
                self.assertEqual(1, parsed.model_execution.unresolved_invalid_records)
                self.assertEqual(0, parsed.model_execution.recovered_invalid_records)
                self.assertEqual([], parsed.directives)

    def test_unknown_and_quoted_patches_never_become_rule_eligible(self):
        raw = {key: value for key, value in fact().items() if key not in {
            "modality", "source_scope", "certainty"
        }}
        for scope, text in (
            ("unknown", "林澈的身份是领航员。"),
            ("quoted_material", "档案记载林澈的身份是领航员。"),
        ):
            with self.subTest(scope=scope):
                parsed, _ = self.run_single(raw, {"patches": [{
                    "record_index": 1,
                    "modality": "asserted",
                    "source_scope": scope,
                    "certainty": "certain",
                }]}, text=text)
                self.assertEqual(1, len(parsed.directives))
                self.assertFalse(eligible_for_deterministic_rules(parsed.directives[0]))
                if scope == "unknown":
                    self.assertNotEqual("character_claim", parsed.directives[0].kind)

    def test_repaired_world_rule_cannot_launder_a_quoted_proposition(self):
        raw = {
            "kind": "world_rule",
            "key": "潮汐术",
            "value": "失效",
            "source_line_start": 1,
            "source_line_end": 1,
        }
        parsed, calls = self.run_single(
            raw,
            {"patches": [{
                "record_index": 1,
                "modality": "asserted",
                "source_scope": "world_rule",
                "certainty": "certain",
            }]},
            text="档案写道：“潮汐术必须失效。”",
        )
        self.assertEqual(2, len(calls))
        self.assertTrue(parsed.model_execution.repair_succeeded)
        self.assertEqual("quoted_material", parsed.directives[0].attrs["source_scope"])
        self.assertFalse(eligible_for_deterministic_rules(parsed.directives[0]))

    def test_failed_repair_salvages_only_explicit_noncanonical_surfaces(self):
        cases = [
            ({"kind": "open_question", "question": "林澈是否是领航员？", "question_type": "open", "source_line_start": 1, "source_line_end": 1}, "林澈是否是领航员？"),
            ({key: value for key, value in fact().items() if key not in {"modality", "source_scope", "certainty"}}, "如果林澈的身份是领航员，也许他会留下。"),
            ({key: value for key, value in fact().items() if key not in {"modality", "source_scope", "certainty"}}, "苏弦说：“林澈的身份是领航员。”"),
            ({"kind": "world_rule", "key": "静默海域", "value": "失效", "source_line_start": 1, "source_line_end": 1}, "如果进入静默海域，潮汐术会失效。"),
            ({"kind": "fact", "subject": "林澈", "predicate": "抵达", "value": "山门", "source_line_start": 1, "source_line_end": 1}, "林澈并非不可能抵达山门。"),
        ]
        for raw, text in cases:
            with self.subTest(text=text):
                parsed, _ = self.run_single(raw, {"patches": []}, text=text)
                self.assertEqual(1, len(parsed.directives))
                self.assertEqual("failed", parsed.directives[0].attrs["repair_status"])
                self.assertEqual(1, parsed.model_execution.unresolved_invalid_records)
                self.assertEqual(0, parsed.model_execution.recovered_invalid_records)
                self.assertFalse(eligible_for_deterministic_rules(parsed.directives[0]))

    def test_context_injection_is_not_repairable(self):
        raw = fact(story_scope="attacker")
        calls = []
        provider = provider_for([{"records": [raw]}], calls)
        parsed = ModelEnhancedExtractor(provider, baseline=EmptyBaseline()).extract(
            DocumentInput("doc", "chapter.md", "林澈的身份是领航员。", "canon", "real")
        )
        self.assertEqual(1, len(calls))
        self.assertFalse(parsed.model_execution.repair_attempted)
        self.assertEqual(1, parsed.model_execution.unresolved_invalid_records)
        self.assertEqual([], parsed.directives)

    def test_repair_budget_cancel_and_timeout_are_bounded(self):
        raw = {key: value for key, value in fact().items() if key not in {
            "modality", "source_scope", "certainty"
        }}
        calls = []
        provider = provider_for([{"records": [raw]}], calls, per_run_token_budget=2300)
        parsed = ModelEnhancedExtractor(provider, baseline=EmptyBaseline()).extract(
            DocumentInput("doc", "chapter.md", "林澈的身份是领航员。")
        )
        self.assertEqual(1, len(calls))
        self.assertEqual("repair_token_budget", parsed.model_execution.repair_skipped_reason)

        calls = []
        provider = provider_for([{"records": [raw]}], calls)
        extractor = ModelEnhancedExtractor(provider, baseline=EmptyBaseline())
        extractor.begin_run(
            checkpoint=lambda: (_ for _ in ()).throw(AnalysisCancelled("cancel"))
            if calls else None
        )
        with self.assertRaises(AnalysisCancelled):
            extractor.extract(DocumentInput("doc", "chapter.md", "林澈的身份是领航员。"))
        self.assertEqual(1, len(calls))

        calls = []
        provider = provider_for(
            [{"records": [raw]}, httpx.ReadTimeout("repair timeout")], calls
        )
        parsed = ModelEnhancedExtractor(provider, baseline=EmptyBaseline()).extract(
            DocumentInput("doc", "chapter.md", "林澈的身份是领航员。")
        )
        self.assertEqual(2, len(calls))
        self.assertTrue(parsed.model_execution.repair_failed)
        self.assertEqual("read_timeout", parsed.model_execution.provider_calls[-1].category)

    def test_later_chunk_run_limit_preserves_success_as_partial_repair(self):
        first = {
            key: value for key, value in fact().items()
            if key not in {"modality", "source_scope", "certainty"}
        }
        second = {
            **first,
            "subject": "顾青",
            "value": "守卫",
            "source_line_start": 2,
            "source_line_end": 2,
        }
        calls = []
        provider = provider_for([
            {"records": [first]},
            {"patches": [{
                "record_index": 1,
                "modality": "asserted",
                "source_scope": "narrator",
                "certainty": "certain",
            }]},
            {"records": [second]},
        ], calls, model_chunk_max_chars=32, model_chunk_overlap_lines=0)
        parsed = ModelEnhancedExtractor(provider, baseline=EmptyBaseline()).extract(
            DocumentInput(
                "doc",
                "chapter.md",
                "正式档案再次明确说明林澈的身份是领航员，内容有效。\n"
                "补充背景内容足够长，顾青的身份是守卫，这是确定事实。",
            )
        )
        repair = parsed.model_execution
        self.assertEqual(3, len(calls))
        self.assertTrue(repair.repair_attempted)
        self.assertTrue(repair.repair_succeeded)
        self.assertFalse(repair.repair_failed)
        self.assertEqual("repair_run_limit", repair.repair_skipped_reason)
        self.assertEqual((2, 1, 0, 1), (
            repair.repair_pre_invalid,
            repair.repair_post_invalid,
            repair.repair_salvaged,
            repair.repair_dropped,
        ))
        self.assertEqual((2, 1, 1), (
            repair.invalid_records,
            repair.unresolved_invalid_records,
            repair.recovered_invalid_records,
        ))
        self.assertEqual("partial_repair", repair.repair_final_path)

    def test_run_aggregation_keeps_repaired_then_baseline_document_order(self):
        first = {
            key: value for key, value in fact().items()
            if key not in {"modality", "source_scope", "certainty"}
        }
        later = {
            **first,
            "subject": "顾青",
            "value": "守卫",
        }
        calls = []
        provider = provider_for([
            {"records": [first]},
            {"patches": [{
                "record_index": 1,
                "modality": "asserted",
                "source_scope": "narrator",
                "certainty": "certain",
            }]},
            {"records": []},
            {"records": [later]},
        ], calls, model_chunk_max_chars=32, model_chunk_overlap_lines=0)
        result = AnalysisPipeline(
            extractor=ModelEnhancedExtractor(provider, baseline=EmptyBaseline())
        ).run([
            DocumentInput(
                "first",
                "first.md",
                "正式档案再次明确说明林澈的身份是领航员，内容有效。\n"
                "第二段背景内容足够长，但没有新的结构化叙事状态记录。",
            ),
            DocumentInput(
                "later",
                "later.md",
                "补充背景内容足够长，顾青的身份是守卫，这是确定事实。",
            ),
        ])
        status = result.diagnostics["model"]
        self.assertEqual(4, len(calls))
        self.assertTrue(status["repair_attempted"])
        self.assertTrue(status["repair_succeeded"])
        self.assertFalse(status["repair_failed"])
        self.assertEqual("repair_run_limit", status["repair_skipped_reason"])
        self.assertEqual((2, 1, 0, 1), (
            status["repair_pre_invalid"],
            status["repair_post_invalid"],
            status["repair_salvaged"],
            status["repair_dropped"],
        ))
        self.assertEqual((2, 1, 1), (
            status["invalid_records"],
            status["unresolved_invalid_records"],
            status["recovered_invalid_records"],
        ))
        self.assertEqual("partial_repair", status["repair_final_path"])
        self.assertEqual("partial_repair", status["repair"]["final_path"])
        self.assertEqual(
            ["repaired", "baseline"],
            [row["repair_final_path"] for row in status["documents"]],
        )

    def test_batch_repair_is_atomic_and_unions_baseline_model_provenance(self):
        raw = {key: value for key, value in fact().items() if key not in {
            "modality", "source_scope", "certainty"
        }}
        calls = []
        provider = provider_for([
            {"documents": [
                {"doc_ref": "d1", "records": [raw]},
                {"doc_ref": "d2", "records": []},
            ]},
            {"patches": [{"record_index": 1, "modality": "asserted", "source_scope": "narrator", "certainty": "certain"}]},
        ], calls)
        result = AnalysisPipeline(extractor=ModelEnhancedExtractor(provider)).run([
            DocumentInput("a", "a.md", "林澈的身份是领航员。"),
            DocumentInput("b", "b.md", "普通背景。"),
        ])
        self.assertEqual(2, len(calls))
        self.assertEqual(2, result.diagnostics["model"]["logical_call_count"])
        self.assertEqual((24, 16), (result.prompt_tokens, result.completion_tokens))
        self.assertEqual((1, 0, 1), (
            result.diagnostics["model"]["invalid_records"],
            result.diagnostics["model"]["unresolved_invalid_records"],
            result.diagnostics["model"]["recovered_invalid_records"],
        ))
        fact_rows = [row for row in result.diagnostics["provenance"]["directives"]]
        self.assertIn(["baseline", "model"], [row["sources"] for row in fact_rows])

    def test_batch_label_repair_survives_content_invalid_sibling_group(self):
        missing_labels = {
            key: value
            for key, value in fact().items()
            if key not in {"modality", "source_scope", "certainty"}
        }
        unsupported = fact(subject="顾青", value="守卫")
        calls = []
        provider = provider_for([
            {"documents": [
                {"doc_ref": "d1", "records": [missing_labels]},
                {"doc_ref": "d2", "records": [unsupported]},
            ]},
            {"patches": [{
                "record_index": 1,
                "modality": "asserted",
                "source_scope": "narrator",
                "certainty": "certain",
            }]},
        ], calls)
        result = AnalysisPipeline(
            extractor=ModelEnhancedExtractor(provider, baseline=EmptyBaseline())
        ).run([
            DocumentInput("a", "a.md", "林澈的身份是领航员。"),
            DocumentInput("b", "b.md", "苏弦的身份是档案官。"),
        ])

        self.assertEqual(2, len(calls))
        status = result.diagnostics["model"]
        self.assertEqual((2, 1, 1), (
            status["invalid_records"],
            status["unresolved_invalid_records"],
            status["recovered_invalid_records"],
        ))
        self.assertTrue(status["repair_succeeded"])
        self.assertEqual((1, 1), (
            status["succeeded_chunks"], status["failed_chunks"]
        ))
        self.assertIn("lexical_support", status["reason_codes"])
        self.assertNotIn("batch_protocol", status["reason_codes"])
        self.assertEqual(["a"], [
            row.evidence.document_id
            for row in result.directives
            if "model" in row.provenance_sources
        ])


if __name__ == "__main__":
    unittest.main()
