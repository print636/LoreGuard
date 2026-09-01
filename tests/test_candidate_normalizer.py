import unittest
from pathlib import Path

from app.domain import EvidenceSpan, ParsedDirective
from app.parser import ParsedDocument
from app.pipeline import AnalysisPipeline, BaselineExtractor, DocumentInput


class CandidateNormalizerTests(unittest.TestCase):
    def test_take_item_then_stamp_or_use_becomes_canonical_use(self):
        documents = [
            DocumentInput(
                id="world",
                name="world.md",
                content="王都保存着唯一一枚“赤曜章”。徽章一直由执政官季遥保管。",
            ),
            DocumentInput(
                id="chapter",
                name="chapter.md",
                content=(
                    "沈舟进入档案室。\n"
                    "他从匣中取出赤曜章，并在没有交接文书的情况下盖下通行印记。"
                ),
            ),
        ]
        result = AnalysisPipeline(extractor=BaselineExtractor()).run(documents)
        records = [row for row in result.directives if row.kind in {"item", "uses"}]
        self.assertEqual(2, len(records))
        self.assertEqual("赤曜章", records[0].attrs["item"])
        self.assertEqual("执政官季遥", records[0].attrs["owner"])
        self.assertEqual("赤曜章", records[1].attrs["item"])
        self.assertEqual("沈舟", records[1].attrs["user"])
        issue = next(row for row in result.issues if row.category.value == "item_ownership")
        self.assertEqual({"world.md", "chapter.md"}, {span.document_name for span in issue.evidence})

    def test_take_item_and_use_wording_is_supported(self):
        document = DocumentInput(
            id="chapter",
            name="chapter.md",
            content="顾青进入库房。\n她拿出银钥匙并使用它打开密柜。",
        )
        result = AnalysisPipeline(extractor=BaselineExtractor()).run([document])
        use = next(row for row in result.directives if row.kind == "uses")
        self.assertEqual("顾青", use.attrs["user"])
        self.assertEqual("银钥匙", use.attrs["item"])
        self.assertEqual(2, use.evidence.line_start)

    def test_performing_disabled_ability_inside_scope_becomes_rule_assertion(self):
        documents = [
            DocumentInput(
                id="world",
                name="world.md",
                content="进入禁鸣峡谷后，所有声波术都会失效。",
            ),
            DocumentInput(
                id="chapter",
                name="chapter.md",
                content="黎安在禁鸣峡谷中央仍发动声波术，震开了石门。",
            ),
        ]
        result = AnalysisPipeline(extractor=BaselineExtractor()).run(documents)
        rule = next(row for row in result.directives if row.kind == "world_rule")
        assertion = next(row for row in result.directives if row.kind == "world_assert")
        self.assertEqual(rule.attrs["key"], assertion.attrs["key"])
        issue = next(row for row in result.issues if row.category.value == "world_rule_conflict")
        self.assertEqual({"world.md", "chapter.md"}, {span.document_name for span in issue.evidence})

    def test_mentions_of_use_are_not_treated_as_item_actions(self):
        document = DocumentInput(
            id="doc",
            name="notes.md",
            content="两份记录使用的是同一套时钟。\n此后他一直使用机械义肢。",
        )
        result = AnalysisPipeline(extractor=BaselineExtractor()).run([document])
        self.assertFalse(any(row.kind == "uses" for row in result.directives))

    def test_advanced_fixture_recovers_the_two_previously_missing_categories(self):
        root = Path(__file__).resolve().parents[1] / "data" / "advanced"
        documents = [
            DocumentInput(
                id=path.stem,
                name=path.name,
                content=path.read_text(encoding="utf-8"),
            )
            for path in (root / "world.md", root / "chapter-01.md", root / "chapter-02.md")
        ]
        result = AnalysisPipeline(extractor=BaselineExtractor()).run(documents)
        counts = {}
        for issue in result.issues:
            counts[issue.category.value] = counts.get(issue.category.value, 0) + 1
        self.assertEqual(
            {
                "fact_conflict": 1,
                "location_collision": 1,
                "knowledge_without_acquisition": 1,
                "item_ownership": 1,
                "world_rule_conflict": 1,
            },
            counts,
        )
        knowledge = next(
            issue for issue in result.issues if issue.category.value == "knowledge_without_acquisition"
        )
        self.assertEqual(2, len(knowledge.evidence))
        self.assertEqual("林澈", knowledge.metadata["character"])
        self.assertEqual("沉星航线入口位置", knowledge.metadata["fact"])
        for issue in result.issues:
            self.assertNotIn("body_state:", issue.title)
            self.assertNotIn("scope_action:", issue.title)

    def test_persistent_body_state_conflict_is_canonicalized(self):
        documents = [
            DocumentInput(
                id="world",
                name="world.md",
                content="程砚在旧日事故中失去了右臂，之后一直使用机械义肢。",
            ),
            DocumentInput(
                id="chapter",
                name="chapter.md",
                content="程砚发现吊索断裂。他立刻伸出完好的右手抓住吊索，手臂没有受伤。",
            ),
        ]
        result = AnalysisPipeline(extractor=BaselineExtractor()).run(documents)
        issue = next(row for row in result.issues if row.category.value == "fact_conflict")
        self.assertEqual("body_state:右上肢", issue.metadata["predicate"])
        self.assertEqual({"world.md", "chapter.md"}, {span.document_name for span in issue.evidence})

    def test_recovery_or_illusory_limb_does_not_create_body_conflict(self):
        recovery = [
            DocumentInput(
                id="world",
                name="world.md",
                content="周岚曾在事故中失去了右臂，但治疗后右臂已经完全再生。",
            ),
            DocumentInput(
                id="chapter",
                name="chapter.md",
                content="周岚进入庭院。她伸出完好的右手接住花瓣。",
            ),
        ]
        illusion = [
            DocumentInput(
                id="world-2",
                name="world-2.md",
                content="孟川在战斗中失去了左臂。",
            ),
            DocumentInput(
                id="chapter-2",
                name="chapter-2.md",
                content="幻象中，孟川伸出看似完好的左手；现实中的身体没有改变。",
            ),
        ]
        for documents in (recovery, illusion):
            result = AnalysisPipeline(extractor=BaselineExtractor()).run(documents)
            self.assertFalse(any(row.category.value == "fact_conflict" for row in result.issues))

    def test_same_item_use_with_different_optional_time_produces_one_issue(self):
        owner_evidence = EvidenceSpan(
            document_id="doc", document_name="doc.md", line_start=1, line_end=1,
            text="赤曜章由季遥保管。",
        )
        use_evidence = EvidenceSpan(
            document_id="doc", document_name="doc.md", line_start=2, line_end=2,
            text="沈舟取出赤曜章并盖下印记。",
        )

        class DuplicateExtractor:
            def extract(self, document):
                return ParsedDocument(
                    document_id=document.id,
                    document_name=document.name,
                    directives=[
                        ParsedDirective(kind="item", attrs={"item": "赤曜章", "owner": "季遥", "time": ""}, evidence=owner_evidence),
                        ParsedDirective(kind="uses", attrs={"item": "赤曜章", "user": "沈舟", "time": "午后"}, evidence=use_evidence),
                        ParsedDirective(kind="uses", attrs={"item": "赤曜章", "user": "沈舟", "time": ""}, evidence=use_evidence),
                    ],
                )

        result = AnalysisPipeline(extractor=DuplicateExtractor()).run(
            [DocumentInput(id="doc", name="doc.md", content="赤曜章由季遥保管。\n沈舟取出赤曜章并盖下印记。")]
        )
        self.assertEqual(1, len([row for row in result.directives if row.kind == "uses"]))
        self.assertEqual(1, len([row for row in result.issues if row.category.value == "item_ownership"]))

    def test_only_applicable_actor_route_and_active_mobility_permission_suppresses_collision(self):
        base_events = "1027-03-05 09:00，叶峤在雾港。\n1027-03-05 09:00，叶峤在山门。"
        valid = "管理局向叶峤签发瞬时通行许可：允许从雾港抵达山门，有效期从1027-03-01 00:00到1027-03-20 18:00，当前有效。"
        invalid_variants = [
            "管理局向温岚签发瞬时通行许可：允许从雾港抵达山门，有效期从1027-03-01 00:00到1027-03-20 18:00，当前有效。",
            "叶峤持有有效的传送通行证，可在1027-03-01 00:00至1027-03-20 18:00期间往返雾港与灯塔。",
            "叶峤曾获从雾港到山门的跃迁许可，但已于1027-03-01 01:00过期。",
            "叶峤的从雾港到山门传送权限当前无效。",
        ]
        result = AnalysisPipeline(extractor=BaselineExtractor()).run([
            DocumentInput(id="permission", name="permission.md", content=valid),
            DocumentInput(id="chapter", name="chapter.md", content=base_events),
        ])
        self.assertFalse(any(row.category.value == "location_collision" for row in result.issues))
        for index, permission in enumerate(invalid_variants):
            result = AnalysisPipeline(extractor=BaselineExtractor()).run([
                DocumentInput(id=f"permission-{index}", name="permission.md", content=permission),
                DocumentInput(id=f"chapter-{index}", name="chapter.md", content=base_events),
            ])
            self.assertTrue(any(row.category.value == "location_collision" for row in result.issues))

    def test_explicit_knowledge_sources_before_claim_are_canonical(self):
        sources = [
            "1027-04-01 08:00，叶峤阅读来信后获知避风航线位置。",
            "1027-04-01 08:00，叶峤亲眼目击并知道了避风航线位置。",
            "1027-04-01 08:00，值守官把避风航线位置告诉叶峤。",
            "1027-04-01 08:00，叶峤查阅航海档案，得知避风航线位置。",
        ]
        claim = "1027-04-01 09:00，叶峤准确说出了避风航线位置。"
        for index, source in enumerate(sources):
            result = AnalysisPipeline(extractor=BaselineExtractor()).run([
                DocumentInput(id=f"knowledge-{index}", name="chapter.md", content=source + "\n" + claim),
            ])
            self.assertFalse(any(row.category.value == "knowledge_without_acquisition" for row in result.issues))
        late = AnalysisPipeline(extractor=BaselineExtractor()).run([
            DocumentInput(id="late", name="chapter.md", content=claim + "\n" + sources[0].replace("08:00", "10:00")),
        ])
        self.assertTrue(any(row.category.value == "knowledge_without_acquisition" for row in late.issues))

    def test_rule_exception_must_match_actor_key_and_active_state(self):
        rule = "在静潮域中，任何回声术都会失效。"
        assertion = "叶峤却在静潮域中发动回声术。"
        valid = "叶峤拥有在静潮域使用回声术的例外资格，当前有效。"
        invalid = [
            "温岚拥有在静潮域使用回声术的例外资格，当前有效。",
            "叶峤拥有在雾港使用回声术的例外资格，当前有效。",
            "叶峤拥有在静潮域使用回声术的例外资格，当前无效。",
        ]
        result = AnalysisPipeline(extractor=BaselineExtractor()).run([
            DocumentInput(id="world", name="world.md", content=rule + "\n" + valid),
            DocumentInput(id="chapter", name="chapter.md", content=assertion),
        ])
        self.assertFalse(any(row.category.value == "world_rule_conflict" for row in result.issues))
        for index, exception in enumerate(invalid):
            result = AnalysisPipeline(extractor=BaselineExtractor()).run([
                DocumentInput(id=f"world-{index}", name="world.md", content=rule + "\n" + exception),
                DocumentInput(id=f"chapter-{index}", name="chapter.md", content=assertion),
            ])
            self.assertTrue(any(row.category.value == "world_rule_conflict" for row in result.issues))

    def test_different_timestamps_are_state_transition_not_fact_conflict(self):
        evidence_one = EvidenceSpan(document_id="state", document_name="state.md", line_start=1, line_end=1, text="旧状态")
        evidence_two = EvidenceSpan(document_id="state", document_name="state.md", line_start=2, line_end=2, text="恢复后状态")

        class TimedFactExtractor:
            def extract(self, document):
                return ParsedDocument(document_id=document.id, document_name=document.name, directives=[
                    ParsedDirective(kind="fact", attrs={"subject": "叶峤", "predicate": "行动能力", "value": "受限", "time": "1027-01-01 08:00"}, evidence=evidence_one),
                    ParsedDirective(kind="fact", attrs={"subject": "叶峤", "predicate": "行动能力", "value": "恢复", "time": "1027-02-01 08:00"}, evidence=evidence_two),
                ])

        result = AnalysisPipeline(extractor=TimedFactExtractor()).run([
            DocumentInput(id="state", name="state.md", content="旧状态\n恢复后状态"),
        ])
        self.assertFalse(any(row.category.value == "fact_conflict" for row in result.issues))

    def test_natural_timed_facts_preserve_state_transition_time(self):
        result = AnalysisPipeline(extractor=BaselineExtractor()).run([
            DocumentInput(
                id="state",
                name="state.md",
                content=(
                    "1027-01-01 08:00，叶峤的行动能力是受限。\n"
                    "1027-02-01 08:00，叶峤的行动能力是恢复。"
                ),
            ),
        ])

        facts = [row for row in result.directives if row.kind == "fact"]
        self.assertEqual(
            ["1027-01-01 08:00", "1027-02-01 08:00"],
            [row.attrs.get("time") for row in facts],
        )
        self.assertFalse(any(row.category.value == "fact_conflict" for row in result.issues))


if __name__ == "__main__":
    unittest.main()
