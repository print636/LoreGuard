import unittest

from app.evaluation import generate_cases, run_evaluation
from app.parser import parse_document
from app.retrieval import HybridRetriever
from app.rules import detect_issues
from app.domain import EvidenceSpan, ParsedDirective


class ParserAndRuleTests(unittest.TestCase):
    def test_natural_chinese_baseline_detects_all_five_categories(self):
        text = """林澈的发色是银色。
林澈的发色是黑色。
1026-04-03 10:00，林澈在北港。
1026-04-03 10:00，林澈在南塔。
1026-04-03 12:00，林澈得知星门口令。
1026-04-03 09:00，林澈说出星门口令。
1026-04-03 08:00，苏弦保管星门钥匙。
1026-04-03 09:30，林澈使用星门钥匙。
星门只能由潮汐晶核驱动。
星门由普通火焰驱动。"""
        parsed = parse_document("natural", "natural.md", text)
        issues = detect_issues(parsed.directives)
        self.assertEqual(5, len({issue.category for issue in issues}))
        self.assertTrue(all(issue.evidence for issue in issues))

    def test_all_five_rules(self):
        text = """@fact subject="林澈" predicate="发色" value="银色" | 设定一
@fact subject="林澈" predicate="发色" value="黑色" | 设定二
@event id="a" time="1026-01-01T10:00" location="北港" participants="林澈" | 北港
@event id="b" time="1026-01-01T10:00" location="南塔" participants="林澈" | 南塔
@claims_knows character="林澈" fact="密码" time="1026-01-01T09:00" | 提前知道
@knows character="林澈" fact="密码" time="1026-01-01T12:00" | 后来得知
@item item="钥匙" owner="苏弦" time="1026-01-01T08:00" | 苏弦持有
@uses item="钥匙" user="林澈" time="1026-01-01T09:00" | 林澈使用
@world_rule key="能源" value="晶核" | 权威规则
@world_assert key="能源" value="火焰" | 冲突章节"""
        parsed = parse_document("doc", "sample.md", text)
        issues = detect_issues(parsed.directives)
        self.assertEqual(5, len(issues))
        self.assertEqual(5, len({issue.category for issue in issues}))
        self.assertTrue(all(issue.evidence for issue in issues))

    def test_json_directives(self):
        content = '{"directives":["@fact subject=甲 predicate=身份 value=船长 | 证据"]}'
        parsed = parse_document("1", "a.json", content)
        self.assertEqual("甲", parsed.directives[0].attrs["subject"])

    def test_hybrid_retrieval(self):
        parsed = parse_document("1", "a.md", '@fact subject="林澈" predicate="发色" value="银色" | 林澈拥有银色头发。\n@fact subject="苏弦" predicate="武器" value="长弓" | 苏弦持有长弓。')
        evidence = [d.evidence for d in parsed.directives]
        result = HybridRetriever().rank("林澈 发色", evidence)
        self.assertIn("林澈", result[0].text)

    def test_reported_timed_location_uses_person_not_reporting_prefix(self):
        text = """1026-06-01 08:30，航海日志显示顾青仍在东港仓库盘点物资。
1026-06-01 08:30，顾青在西塔顶层与站长会面。"""
        parsed = parse_document("doc", "chapter.md", text)
        events = [row for row in parsed.directives if row.kind == "event"]
        self.assertEqual(2, len(events))
        self.assertTrue(all(row.attrs["participants"] == "顾青" for row in events))
        self.assertEqual({"东港仓库", "西塔顶层"}, {row.attrs["location"] for row in events})

    def test_location_collision_requires_precise_time_and_distinct_evidence(self):
        fuzzy = parse_document(
            "fuzzy",
            "chapter.md",
            '@event time="午后" location="北港" participants="林澈" | 北港\n'
            '@event time="午后" location="南塔" participants="林澈" | 南塔',
        )
        self.assertFalse(any(issue.category.value == "location_collision" for issue in detect_issues(fuzzy.directives)))

        evidence = EvidenceSpan(
            document_id="same",
            document_name="same.md",
            line_start=1,
            line_end=1,
            text="同一句模型抽取产生两个地点候选。",
        )
        duplicated_span = [
            ParsedDirective(kind="event", attrs={"time": "1026-01-01 10:00", "location": location, "participants": "林澈"}, evidence=evidence)
            for location in ("北港", "南塔")
        ]
        self.assertFalse(any(issue.category.value == "location_collision" for issue in detect_issues(duplicated_span)))

        nested = parse_document(
            "nested",
            "chapter.md",
            '@event time="1026-01-01 10:00" location="北港议会" participants="林澈" | 林澈返回北港议会并走入密室。\n'
            '@event time="1026-01-01 10:00" location="密室" participants="林澈" | 林澈在北港议会的密室见到苏弦。',
        )
        self.assertFalse(any(issue.category.value == "location_collision" for issue in detect_issues(nested.directives)))

    def test_knowledge_issue_requires_explicit_later_acquisition(self):
        claim_only = parse_document(
            "claim",
            "chapter.md",
            '@claims_knows character="林澈" fact="航线入口" time="1026-01-01 09:00" | 林澈说出入口。',
        )
        self.assertFalse(any(issue.category.value == "knowledge_without_acquisition" for issue in detect_issues(claim_only.directives)))

        same_evidence = EvidenceSpan(
            document_id="knowledge",
            document_name="knowledge.md",
            line_start=1,
            line_end=1,
            text="此刻苏弦第一次告诉林澈入口，句中也提到他此前不知道。",
        )
        directives = [
            ParsedDirective(kind="claims_knows", attrs={"character": "林澈", "fact": "航线入口", "time": "此前"}, evidence=same_evidence),
            ParsedDirective(kind="knows", attrs={"character": "林澈", "fact": "航线入口", "time": "1026-01-01 10:00"}, evidence=same_evidence),
        ]
        self.assertFalse(any(issue.category.value == "knowledge_without_acquisition" for issue in detect_issues(directives)))


class EvaluationTests(unittest.TestCase):
    def test_dataset_has_eighty_independent_cases(self):
        cases = generate_cases()
        self.assertEqual(80, len(cases))
        self.assertEqual(80, len({case["id"] for case in cases}))

    def test_directive_regression_meets_rule_wiring_gate(self):
        result = run_evaluation()
        self.assertGreaterEqual(result.precision, 0.75)
        self.assertGreaterEqual(result.recall, 0.60)
        self.assertGreaterEqual(result.evidence_hit_rate, 0.85)


if __name__ == "__main__":
    unittest.main()
