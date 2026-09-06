import json
from types import SimpleNamespace
import unittest

import httpx

from app.domain import ConsistencyIssue, EvidenceSpan, IssueCategory, ParsedDirective, Severity
from app.model_extractor import ModelEnhancedExtractor, merge_directives
from app.parser import ParsedDocument
from app.pipeline import AnalysisPipeline, BaselineExtractor, DocumentInput
from app.provider import OpenAICompatibleProvider, RetryPolicy
from scripts.run_phase1_model_acceptance import _directive_fingerprint, _issue_fingerprint
from tests.test_model_extractor import completion, settings


def fact(subject: str, value: str, line: int = 1) -> dict:
    return {
        "kind": "fact",
        "subject": subject,
        "predicate": "身份",
        "value": value,
        "source_line_start": line,
        "source_line_end": line,
        "modality": "asserted",
        "source_scope": "narrator",
        "certainty": "certain",
    }


class EmptyBaseline:
    def extract(self, document):
        return ParsedDocument(document_id=document.id, document_name=document.name)


class ProvenanceTests(unittest.TestCase):
    def provider(self, payload: dict) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            settings(model_batch_max_chars=20_000),
            transport=httpx.MockTransport(
                lambda _: completion(json.dumps(payload, ensure_ascii=False))
            ),
            retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0),
            sleep=lambda _: None,
        )

    def test_baseline_only_has_stable_runner_compatible_fingerprint(self):
        document = DocumentInput(
            "case:fixtures/chapter.md",
            "chapter.md",
            '@fact subject="林澈" predicate="身份" value="领航员" | 设定',
        )
        result = AnalysisPipeline(extractor=BaselineExtractor()).run([document])
        provenance = result.diagnostics["provenance"]
        self.assertEqual(1, provenance["schema_version"])
        self.assertEqual(["baseline"], provenance["directives"][0]["sources"])
        self.assertEqual(
            _directive_fingerprint(result.directives[0], "fixtures/chapter.md"),
            provenance["directives"][0]["fingerprint"],
        )
        self.assertNotIn("provenance_sources", result.directives[0].model_dump())

    def test_single_document_model_only_does_not_spread_to_baseline_rows(self):
        document = DocumentInput(
            "single",
            "single.md",
            "林澈的身份是领航员。\n@fact subject=苏弦 predicate=身份 value=档案官 | 设定",
        )
        extractor = ModelEnhancedExtractor(
            self.provider({"records": [fact("林澈", "领航员", 1)]}),
            baseline=EmptyBaseline(),
        )
        result = AnalysisPipeline(extractor=extractor).run([document])
        rows = result.diagnostics["provenance"]["directives"]
        self.assertEqual([["model"]], [row["sources"] for row in rows])
        self.assertTrue(result.model_used)

        mixed = AnalysisPipeline(extractor=ModelEnhancedExtractor(
            self.provider({"records": [fact("林澈", "领航员", 1)]})
        )).run([document])
        sources_by_subject = {
            directive.attrs.get("subject"): row["sources"]
            for directive, row in self._directive_rows(mixed)
        }
        self.assertEqual(["baseline"], sources_by_subject["苏弦"])
        self.assertEqual(["baseline", "model"], sources_by_subject["林澈"])

    def test_duplicate_baseline_and_model_directive_unions_sources(self):
        evidence = EvidenceSpan(
            document_id="doc",
            document_name="chapter.md",
            line_start=1,
            line_end=1,
            text="林澈的身份是领航员。",
        )
        attrs = {
            "subject": "林澈",
            "predicate": "身份",
            "value": "领航员",
            "modality": "asserted",
            "source_scope": "narrator",
            "certainty": "certain",
        }
        baseline = ParsedDirective(
            kind="fact", attrs=attrs, evidence=evidence,
            provenance_sources=frozenset({"baseline"}),
        )
        model = ParsedDirective(
            kind="fact", attrs=attrs, evidence=evidence,
            provenance_sources=frozenset({"model"}),
        )
        merged = merge_directives([baseline], [model])
        self.assertEqual(1, len(merged))
        self.assertEqual(
            frozenset({"baseline", "model"}), merged[0].provenance_sources
        )

    def test_semantic_gate_convergence_unions_instead_of_losing_model_source(self):
        text = "推车食物是否等同于餐车提供的食物？"

        class UnsafeBaseline:
            def extract(self, document):
                return ParsedDocument(
                    document_id=document.id,
                    document_name=document.name,
                    directives=[ParsedDirective(
                        kind="fact",
                        attrs={
                            "subject": "推车食物", "predicate": "等同于",
                            "value": "餐车提供的食物",
                        },
                        evidence=EvidenceSpan(
                            document_id=document.id, document_name=document.name,
                            line_start=1, line_end=1, text=text,
                        ),
                    )],
                )

        payload = {"records": [{
            "kind": "open_question",
            "question": text,
            "question_type": "open",
            "source_line_start": 1,
            "source_line_end": 1,
            "modality": "interrogative",
            "source_scope": "narrator",
            "certainty": "unknown",
        }]}
        result = AnalysisPipeline(extractor=ModelEnhancedExtractor(
            self.provider(payload), baseline=UnsafeBaseline()
        )).run([DocumentInput("doc", "chapter.md", text)])
        questions = [row for row in self._directive_rows(result) if row[0].kind == "open_question"]
        self.assertEqual(1, len(questions))
        self.assertEqual(["baseline", "model"], questions[0][1]["sources"])

    def test_batch_model_only_rows_are_attributed_per_document(self):
        payload = {
            "documents": [
                {"doc_ref": "d2", "records": [fact("苏弦", "档案官")]},
                {"doc_ref": "d1", "records": [fact("林澈", "领航员")]},
            ]
        }
        documents = [
            DocumentInput("a", "a.md", "林澈的身份是领航员。"),
            DocumentInput("b", "b.md", "苏弦的身份是档案官。"),
        ]
        result = AnalysisPipeline(extractor=ModelEnhancedExtractor(
            self.provider(payload), baseline=EmptyBaseline()
        )).run(documents)
        self.assertTrue(result.diagnostics["model"]["batch_used"])
        self.assertEqual(
            [["model"], ["model"]],
            [row["sources"] for row in result.diagnostics["provenance"]["directives"]],
        )

    def test_issue_reports_mixed_evidence_without_content_or_transport_data(self):
        first = self._directive("a", "a.md", 1, "甲", "白", "baseline")
        second = self._directive("b", "b.md", 2, "甲", "黑", "model")
        issue = ConsistencyIssue(
            category=IssueCategory.fact_conflict,
            severity=Severity.high,
            confidence=1,
            title="冲突",
            explanation="冲突",
            evidence=[first.evidence, second.evidence],
            suggestion="检查",
        )

        class Extractor:
            def extract(self, document):
                directive = first if document.id == "a" else second
                return ParsedDocument(
                    document_id=document.id,
                    document_name=document.name,
                    directives=[directive],
                )

        class Checker:
            def check(self, directives):
                return [issue]

        class NoopNormalizer:
            def enrich(self, documents, directives):
                return SimpleNamespace(directives=directives, warnings=[])

        class NoopRetriever:
            def candidate_pairs(self, directives):
                return []

        documents = [
            DocumentInput("a", "a.md", "private baseline evidence"),
            DocumentInput("b", "b.md", "private model evidence"),
        ]
        result = AnalysisPipeline(
            extractor=Extractor(), checker=Checker(), normalizer=NoopNormalizer(),
            retriever=NoopRetriever(),
        ).run(documents)
        row = result.diagnostics["provenance"]["issues"][0]
        self.assertEqual(["deterministic"], row["derivation"])
        self.assertEqual("deterministic_rule", row["derivation_type"])
        self.assertEqual(["baseline", "model"], row["evidence_sources"])
        self.assertEqual(
            [["baseline"], ["model"]],
            [item["sources"] for item in row["contributing_evidence"]],
        )
        self.assertEqual(row["contributing_evidence"], row["evidence_source_details"])
        self.assertEqual(_issue_fingerprint(issue, {"a": "a.md", "b": "b.md"}), row["fingerprint"])
        serialized = json.dumps(result.diagnostics["provenance"], ensure_ascii=False)
        for forbidden in ("private baseline evidence", "private model evidence", "prompt", "api_key", "http"):
            self.assertNotIn(forbidden, serialized)

    def test_multiline_clarification_lists_every_contributing_line(self):
        directive = ParsedDirective(
            kind="clarification",
            attrs={
                "summary": "抵达路径需要确认",
                "category": "missing_causal_bridge",
                "modality": "uncertain",
                "source_scope": "narrator",
                "certainty": "unknown",
            },
            evidence=EvidenceSpan(
                document_id="doc", document_name="chapter.md",
                line_start=2, line_end=3, text="路径不明。\n需要确认。",
            ),
            provenance_sources=frozenset({"model"}),
        )
        provenance = self._static_pipeline(directive).diagnostics["provenance"]
        row = provenance["directives"][0]
        self.assertEqual([["chapter.md", 2], ["chapter.md", 3]], row["contributing_evidence"])
        self.assertEqual(["model"], row["sources"])
        self.assertEqual(
            [["model"], ["model"]],
            [item["sources"] for item in row["contributing_evidence_sources"]],
        )

    @staticmethod
    def _directive(document_id, name, line, subject, value, source):
        return ParsedDirective(
            kind="fact",
            attrs={
                "subject": subject, "predicate": "颜色", "value": value,
                "modality": "asserted", "source_scope": "narrator",
                "certainty": "certain",
            },
            evidence=EvidenceSpan(
                document_id=document_id, document_name=name,
                line_start=line, line_end=line, text=f"{subject}是{value}",
            ),
            provenance_sources=frozenset({source}),
        )

    @staticmethod
    def _directive_rows(result):
        provenance = {
            row["fingerprint"]: row
            for row in result.diagnostics["provenance"]["directives"]
        }
        from scripts.run_phase1_model_acceptance import _directive_fingerprint
        return [
            (
                directive,
                provenance[_directive_fingerprint(directive, directive.evidence.document_name)],
            )
            for directive in result.directives
        ]

    @staticmethod
    def _static_pipeline(directive):
        class Extractor:
            def extract(self, document):
                return ParsedDocument(
                    document_id=document.id,
                    document_name=document.name,
                    directives=[directive],
                )

        class NoopNormalizer:
            def enrich(self, documents, directives):
                return SimpleNamespace(directives=directives, warnings=[])

        class NoopRetriever:
            def candidate_pairs(self, directives):
                return []

        return AnalysisPipeline(
            extractor=Extractor(), normalizer=NoopNormalizer(), retriever=NoopRetriever()
        ).run([DocumentInput("doc", "chapter.md", "line 1\nline 2\nline 3")])


if __name__ == "__main__":
    unittest.main()
