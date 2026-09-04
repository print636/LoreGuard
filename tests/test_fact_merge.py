import json
import unittest

import httpx

from app.domain import EvidenceSpan, ParsedDirective
from app.model_extractor import ModelEnhancedExtractor, merge_directives
from app.pipeline import AnalysisPipeline, DocumentInput
from app.provider import OpenAICompatibleProvider
from tests.test_model_extractor import completion, settings


class FactMergeTests(unittest.TestCase):
    def fact(self, predicate, *, subject="角色甲", value="守卫", time="", line=1):
        return ParsedDirective(
            kind="fact", attrs=dict(subject=subject, predicate=predicate, value=value, time=time),
            evidence=EvidenceSpan(document_id="doc", document_name="chapter.md",
                                  line_start=line, line_end=line, text="同一行原文中的多个事实。"),
        )

    def test_only_baseline_supported_extra_copula_is_merged(self):
        for predicate in ("职务", "所属阵营", "船体材质"):
            for copula in ("是", "为"):
                with self.subTest(predicate=predicate, copula=copula):
                    baseline = self.fact(predicate)
                    model = self.fact(predicate + copula)
                    merged = merge_directives([baseline], [model])
                    self.assertEqual([baseline], merged)
                    self.assertEqual(predicate + copula, model.attrs["predicate"])

    def test_other_predicate_same_value_same_evidence_is_preserved(self):
        baseline = self.fact("职务")
        model = self.fact("兼职为")
        merged = merge_directives([baseline], [model])
        self.assertEqual(2, len(merged))
        self.assertEqual("兼职为", merged[1].attrs["predicate"])
        # A meaningful word ending in 为 has no independently matching stem.
        self.assertEqual("行为", merge_directives([], [self.fact("行为")])[0].attrs["predicate"])

    def test_different_subject_value_time_and_evidence_do_not_align(self):
        baseline = self.fact("职务")
        for overrides in (
            {"subject": "角色乙"}, {"value": "船长"},
            {"time": "1044-01-01 09:00"}, {"line": 2},
        ):
            with self.subTest(overrides=overrides):
                model = self.fact("职务是", **overrides)
                merged = merge_directives([baseline], [model])
                self.assertEqual(2, len(merged))
                self.assertEqual("职务是", merged[1].attrs["predicate"])

    def test_model_extra_copula_does_not_duplicate_pipeline_conflict(self):
        payload = {"records": [
            {"kind": "fact", "subject": "林澈", "predicate": "身份是", "value": value,
             "source_line_start": index, "source_line_end": index}
            for index, value in enumerate(("领航员", "档案官"), start=1)
        ]}
        provider = OpenAICompatibleProvider(
            settings(), transport=httpx.MockTransport(
                lambda _: completion(json.dumps(payload, ensure_ascii=False))
            ),
        )
        result = AnalysisPipeline(extractor=ModelEnhancedExtractor(provider)).run([
            DocumentInput("doc", "chapter.md", "林澈的身份是领航员。\n林澈的身份是档案官。")
        ])
        self.assertEqual(2, len(result.directives))
        self.assertEqual(1, len(result.issues))
        self.assertEqual("fact_conflict", result.issues[0].category.value)
        self.assertEqual(2, len(result.issues[0].evidence))


if __name__ == "__main__":
    unittest.main()
