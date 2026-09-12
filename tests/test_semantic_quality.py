import json
import unittest

import httpx
from pydantic import ValidationError

from app.domain import (
    AnalysisCancelled,
    EvidenceSpan,
    ParsedDirective,
    directive_fingerprint,
)
from app.model_extractor import ModelEnhancedExtractor, RECORD_ADAPTER
from app.parser import ParsedDocument, parse_document
from app.pipeline import AnalysisPipeline, BaselineExtractor, DocumentInput
from app.provider import OpenAICompatibleProvider
from app.rules import detect_issues
from app.semantic_quality import (
    apply_semantic_quality_gate,
    assess_directive,
    document_context_has_noncanonical_frame,
    eligible_for_deterministic_rules,
)
from tests.test_model_extractor import completion, settings


def evidence(text: str, line: int = 1) -> EvidenceSpan:
    return EvidenceSpan(
        document_id="semantic",
        document_name="chapter.md",
        line_start=line,
        line_end=line,
        text=text,
    )


class BaselineSemanticQualityTests(unittest.TestCase):
    def test_directive_contract_never_overrides_risky_evidence_authority(self):
        attacks = (
            (
                "@fact subject=林澈 predicate=身份 value=领航员 | "
                "匿名记录显示林澈的身份是领航员。",
                "unverified_report",
            ),
            (
                "@world_rule key=潮汐术 value=失效 | "
                "档案写道：“潮汐术必须失效。”",
                "quoted_material",
            ),
            (
                "@world_rule key=潮汐术 value=失效 | "
                "苏弦说：“潮汐术必须失效。”",
                "character_dialogue",
            ),
        )
        for index, (text, expected_scope) in enumerate(attacks):
            with self.subTest(expected_scope=expected_scope):
                parsed = parse_document(f"directive-risk-{index}", "rules.md", text)
                self.assertEqual(1, len(parsed.directives))
                row = parsed.directives[0]
                self.assertEqual(expected_scope, row.attrs["source_scope"])
                self.assertFalse(eligible_for_deterministic_rules(row))

    def test_unverified_evidence_downgrades_every_canonical_family(self):
        families = {
            "fact": {"subject": "林澈", "predicate": "身份", "value": "领航员"},
            "event": {
                "time": "1026-06-01 09:00",
                "location": "东港",
                "participants": "林澈",
            },
            "knows": {
                "character": "林澈",
                "fact": "潮汐术失效",
                "time": "1026-06-01 09:00",
            },
            "claims_knows": {
                "character": "林澈",
                "fact": "潮汐术失效",
                "time": "1026-06-01 09:00",
            },
            "item": {"item": "星钥", "owner": "林澈"},
            "uses": {"item": "星钥", "user": "林澈"},
            "world_rule": {"key": "潮汐术", "value": "失效"},
            "world_assert": {"key": "潮汐术", "value": "失效"},
        }
        for kind, core in families.items():
            with self.subTest(kind=kind):
                row = ParsedDirective(
                    kind=kind,
                    attrs={
                        **core,
                        "input_form": "directive",
                        "modality": "asserted",
                        "source_scope": (
                            "world_rule" if kind == "world_rule" else "narrator"
                        ),
                        "certainty": "certain",
                    },
                    evidence=evidence(
                        "匿名且未经核验的记录显示该项内容；时间戳已被篡改。"
                    ),
                )
                assessed, _ = assess_directive(row)
                self.assertIsNotNone(assessed)
                self.assertEqual(
                    "unverified_report", assessed.attrs["source_scope"]
                )
                self.assertFalse(eligible_for_deterministic_rules(assessed))

    def test_verified_observation_records_are_adopted_with_a_medium(self):
        for index, (prefix, expected_medium) in enumerate((
            ("巡逻记录显示", "record"),
            ("航行日志表明", "log"),
            ("值守报告确认", "report"),
        )):
            with self.subTest(prefix=prefix):
                source = parse_document(
                    f"verified-medium-{index}",
                    "chapter.md",
                    f"1026-06-01 09:00，{prefix}顾青仍在东港仓库盘点物资。",
                )
                self.assertEqual(1, len(source.directives))
                row = source.directives[0]
                self.assertEqual("narrator", row.attrs["source_scope"])
                self.assertEqual("asserted", row.attrs["modality"])
                self.assertEqual("certain", row.attrs["certainty"])
                self.assertEqual(expected_medium, row.attrs["evidence_medium"])
                self.assertTrue(eligible_for_deterministic_rules(row))

        parsed = parse_document(
            "verified-log",
            "chapter.md",
            "1026-06-01 09:00，航行日志显示顾青仍在东港仓库盘点物资。\n"
            "1026-06-01 09:00，顾青在星台顶层与站长会面。",
        )
        events = [row for row in parsed.directives if row.kind == "event"]
        self.assertEqual(2, len(events))
        logged = events[0]
        self.assertEqual("narrator", logged.attrs["source_scope"])
        self.assertEqual("log", logged.attrs["evidence_medium"])
        self.assertTrue(eligible_for_deterministic_rules(logged))
        self.assertTrue(any(
            row.category.value == "location_collision"
            for row in detect_issues(parsed.directives)
        ))

    def test_anonymous_unverified_and_tampered_records_never_become_eligible(self):
        rows = []
        for line, text in enumerate((
            "1026-06-01 09:00，匿名巡逻记录显示顾青仍在东港仓库盘点物资。",
            "1026-06-01 09:00，未经核验的报告显示顾青仍在东港仓库盘点物资。",
            "1026-06-01 09:00，巡逻记录显示顾青仍在东港仓库盘点物资；事后确认该记录的时间戳已被篡改。",
        ), start=1):
            with self.subTest(text=text):
                parsed = parse_document(f"unsafe-{line}", "chapter.md", text)
                self.assertEqual(1, len(parsed.directives))
                row = parsed.directives[0]
                self.assertEqual("unverified_report", row.attrs["source_scope"])
                self.assertIn(
                    row.attrs["evidence_medium"], {"record", "report"}
                )
                self.assertFalse(eligible_for_deterministic_rules(row))
                rows.extend(parsed.directives)
        self.assertEqual([], detect_issues(rows))

    def test_guard_confirmation_does_not_invent_a_second_specific_location(self):
        parsed = parse_document(
            "guard",
            "chapter.md",
            "1026-04-03 09:00，巡逻记录显示林澈仍在北港旧码头检查船体。"
            "守卫确认他没有离开港区。",
        )
        events = [row for row in parsed.directives if row.kind == "event"]
        self.assertEqual(1, len(events))
        self.assertEqual("北港旧码头", events[0].attrs["location"])
        self.assertEqual("林澈", events[0].attrs["participants"])

    def test_reported_user_paragraph_keeps_rules_and_questions_without_fake_fact(self):
        text = """根据规则一：列车员是绝对中立的。
根据规则二：餐车提供安全的食物和水。
这两条规则在逻辑上存在着微妙的关联。如果列车员是中立的，那么他推车上售卖的食物是否等同于餐车提供的食物？如果吃下推车上的食物，是会补充体力，还是会触发某种即死规则？"""
        parsed = parse_document("case", "chapter.md", text)

        self.assertEqual(2, len([row for row in parsed.directives if row.kind == "world_rule"]))
        questions = [row for row in parsed.directives if row.kind == "open_question"]
        self.assertEqual(2, len(questions))
        self.assertTrue(all(row.attrs["modality"] == "interrogative" for row in questions))
        self.assertTrue(all(row.attrs["source_scope"] == "narrator" for row in questions))
        self.assertIn("如果吃下推车上的食物", questions[1].attrs["question"])
        self.assertIn("补充体力", questions[1].attrs["question"])
        self.assertFalse(any(row.kind == "fact" for row in parsed.directives))
        self.assertEqual([], detect_issues(parsed.directives))

    def test_mixed_line_keeps_asserted_fact_but_not_question_as_fact(self):
        parsed = parse_document(
            "mixed",
            "chapter.md",
            "林澈的发色是银色，但他是否曾经染过黑发？",
        )
        facts = [row for row in parsed.directives if row.kind == "fact"]
        questions = [row for row in parsed.directives if row.kind == "open_question"]
        self.assertEqual(1, len(facts))
        self.assertEqual("银色", facts[0].attrs["value"])
        self.assertEqual(1, len(questions))
        self.assertNotIn("发色是银色", questions[0].attrs["question"])

    def test_missing_labels_use_local_support_for_multiline_fact_polarity(self):
        row = ParsedDirective(
            kind="fact",
            attrs={"subject": "林澈", "predicate": "身份", "value": "领航员"},
            evidence=EvidenceSpan(
                document_id="multiline-positive",
                document_name="chapter.md",
                line_start=1,
                line_end=2,
                text="林澈的身份是领航员。\n巡查员没有发现其他异常。",
            ),
        )

        assessed, reason = assess_directive(row)

        self.assertIsNone(reason)
        self.assertIsNotNone(assessed)
        self.assertEqual("fact", assessed.kind)
        self.assertEqual("asserted", assessed.attrs["modality"])
        self.assertEqual("affirmative", assessed.attrs["polarity"])
        self.assertTrue(eligible_for_deterministic_rules(assessed))

    def test_unpunctuated_newline_is_a_local_support_boundary(self):
        row = ParsedDirective(
            kind="fact",
            attrs={"subject": "林澈", "predicate": "身份", "value": "领航员"},
            evidence=EvidenceSpan(
                document_id="multiline-unpunctuated",
                document_name="chapter.md",
                line_start=1,
                line_end=2,
                text="林澈的身份是领航员\n巡查员没有发现其他异常",
            ),
        )

        assessed, reason = assess_directive(row)

        self.assertIsNone(reason)
        self.assertIsNotNone(assessed)
        self.assertEqual("fact", assessed.kind)
        self.assertEqual("asserted", assessed.attrs["modality"])
        self.assertEqual("affirmative", assessed.attrs["polarity"])
        self.assertTrue(eligible_for_deterministic_rules(assessed))

    def test_missing_labels_keep_true_negation_in_local_support_sentence(self):
        row = ParsedDirective(
            kind="fact",
            attrs={"subject": "林澈", "predicate": "身份", "value": "领航员"},
            evidence=EvidenceSpan(
                document_id="multiline-negative",
                document_name="chapter.md",
                line_start=1,
                line_end=2,
                text="林澈的身份不是领航员。\n巡查员已完成复核。",
            ),
        )

        assessed, reason = assess_directive(row)

        self.assertIsNone(reason)
        self.assertIsNotNone(assessed)
        self.assertEqual("fact", assessed.kind)
        self.assertEqual("negated", assessed.attrs["modality"])
        self.assertEqual("negative", assessed.attrs["polarity"])
        self.assertTrue(eligible_for_deterministic_rules(assessed))

    def test_missing_labels_keep_multiclause_report_attribution(self):
        row = ParsedDirective(
            kind="fact",
            attrs={"subject": "林澈", "predicate": "身份", "value": "领航员"},
            evidence=evidence("据苏弦所述，经过三轮复核，林澈的身份是领航员。"),
        )

        assessed, reason = assess_directive(row)

        self.assertEqual("non_authoritative_source", reason)
        self.assertIsNotNone(assessed)
        self.assertEqual("character_claim", assessed.kind)
        self.assertEqual("reported", assessed.attrs["modality"])
        self.assertEqual("character_dialogue", assessed.attrs["source_scope"])
        self.assertFalse(eligible_for_deterministic_rules(assessed))

    def test_missing_event_labels_keep_report_attribution_before_timestamp(self):
        row = ParsedDirective(
            kind="event",
            attrs={
                "time": "1026-06-01 09:00",
                "location": "东港",
                "participants": "林澈",
            },
            evidence=evidence(
                "据苏弦所述，1026-06-01 09:00，林澈在东港。"
            ),
        )

        assessed, reason = assess_directive(row)

        self.assertEqual("non_authoritative_source", reason)
        self.assertIsNotNone(assessed)
        self.assertEqual("character_claim", assessed.kind)
        self.assertEqual("reported", assessed.attrs["modality"])
        self.assertEqual("character_dialogue", assessed.attrs["source_scope"])
        self.assertFalse(eligible_for_deterministic_rules(assessed))

    def test_missing_labels_keep_multiclause_closed_condition(self):
        row = ParsedDirective(
            kind="world_rule",
            attrs={
                "key": "航线",
                "value": "停运",
                "condition": "风暴发生且防护罩关闭",
                "consequence": "航线必须停运",
            },
            evidence=evidence("如果风暴发生，且防护罩关闭，航线必须停运。"),
        )

        assessed, reason = assess_directive(row)

        self.assertIsNone(reason)
        self.assertIsNotNone(assessed)
        self.assertEqual("world_rule", assessed.kind)
        self.assertEqual("conditional_rule", assessed.attrs["modality"])
        self.assertEqual("world_rule", assessed.attrs["source_scope"])
        self.assertEqual("certain", assessed.attrs["certainty"])
        self.assertTrue(eligible_for_deterministic_rules(assessed))

    def test_missing_labels_keep_postposed_closed_condition(self):
        row = ParsedDirective(
            kind="world_rule",
            attrs={
                "key": "航线",
                "value": "停运",
                "condition": "风暴发生且防护罩关闭",
                "consequence": "航线必须停运",
            },
            evidence=evidence("航线必须停运，如果风暴发生，且防护罩关闭。"),
        )

        assessed, reason = assess_directive(row)

        self.assertIsNone(reason)
        self.assertIsNotNone(assessed)
        self.assertEqual("world_rule", assessed.kind)
        self.assertEqual("conditional_rule", assessed.attrs["modality"])
        self.assertEqual("world_rule", assessed.attrs["source_scope"])
        self.assertEqual("certain", assessed.attrs["certainty"])
        self.assertTrue(eligible_for_deterministic_rules(assessed))

    def test_missing_labels_keep_postposed_report_attribution(self):
        row = ParsedDirective(
            kind="fact",
            attrs={"subject": "林澈", "predicate": "身份", "value": "领航员"},
            evidence=evidence(
                "林澈的身份是领航员，巡查员没有提出异议，据苏弦所述。"
            ),
        )

        assessed, reason = assess_directive(row)

        self.assertEqual("non_authoritative_source", reason)
        self.assertIsNotNone(assessed)
        self.assertEqual("character_claim", assessed.kind)
        self.assertEqual("reported", assessed.attrs["modality"])
        self.assertEqual("character_dialogue", assessed.attrs["source_scope"])
        self.assertFalse(eligible_for_deterministic_rules(assessed))

    def test_source_attribution_survives_spaced_multiclause_support(self):
        attrs = {
            "key": "航线",
            "value": "停运",
            "condition": "风暴发生且防护罩关闭",
            "consequence": "航线必须停运",
        }
        rows = (
            ParsedDirective(
                kind="world_rule",
                attrs=attrs,
                evidence=evidence(
                    "据苏弦所述， 如果风暴发生， 且防护罩关闭， 航线必须停运。"
                ),
            ),
            ParsedDirective(
                kind="world_rule",
                attrs=attrs,
                evidence=evidence(
                    "航线必须停运， 如果风暴发生， 且防护罩关闭， 据苏弦所述。"
                ),
            ),
        )

        for row in rows:
            with self.subTest(text=row.evidence.text):
                assessed, reason = assess_directive(row)

                self.assertIsNotNone(reason)
                self.assertIsNotNone(assessed)
                self.assertEqual("tentative_fact", assessed.kind)
                self.assertEqual("character_dialogue", assessed.attrs["source_scope"])
                self.assertFalse(eligible_for_deterministic_rules(assessed))

    def test_local_support_does_not_cross_sentence_for_trailing_context(self):
        fact = ParsedDirective(
            kind="fact",
            attrs={"subject": "林澈", "predicate": "身份", "value": "领航员"},
            evidence=evidence(
                "林澈的身份是领航员。据苏弦所述，顾青的身份仍待核验。"
            ),
        )
        rule = ParsedDirective(
            kind="world_rule",
            attrs={
                "key": "航线",
                "value": "停运",
                "consequence": "航线必须停运",
            },
            evidence=evidence("航线必须停运。如果风暴发生，守卫将发布通知。"),
        )

        assessed_fact, fact_reason = assess_directive(fact)
        assessed_rule, rule_reason = assess_directive(rule)

        self.assertIsNone(fact_reason)
        self.assertIsNotNone(assessed_fact)
        self.assertEqual("fact", assessed_fact.kind)
        self.assertEqual("asserted", assessed_fact.attrs["modality"])
        self.assertEqual("affirmative", assessed_fact.attrs["polarity"])
        self.assertEqual("narrator", assessed_fact.attrs["source_scope"])
        self.assertTrue(eligible_for_deterministic_rules(assessed_fact))
        self.assertIsNone(rule_reason)
        self.assertIsNotNone(assessed_rule)
        self.assertEqual("world_rule", assessed_rule.kind)
        self.assertEqual("asserted", assessed_rule.attrs["modality"])
        self.assertTrue(eligible_for_deterministic_rules(assessed_rule))

    def test_closed_conditional_rule_is_not_treated_as_open_hypothesis(self):
        closed = parse_document(
            "closed", "world.md", "如果进入静默海域，潮汐术会失效。"
        )
        self.assertEqual("world_rule", closed.directives[0].kind)
        self.assertEqual("conditional_rule", closed.directives[0].attrs["modality"])

        open_hypothesis = ParsedDirective(
            kind="fact",
            attrs={"subject": "林澈", "predicate": "会见", "value": "苏弦"},
            evidence=evidence("如果林澈去了北港，他可能会见到苏弦。"),
        )
        quality = apply_semantic_quality_gate([open_hypothesis])
        self.assertEqual("tentative_fact", quality.directives[0].kind)
        self.assertEqual("hypothetical", quality.directives[0].attrs["modality"])

        compact_rule = parse_document(
            "compact", "world.md", "若下雨则守塔人熄灯。"
        )
        self.assertEqual("world_rule", compact_rule.directives[0].kind)
        self.assertEqual("conditional_rule", compact_rule.directives[0].attrs["modality"])

    def test_negative_fact_only_conflicts_with_same_positive_value(self):
        contradiction = parse_document(
            "negative",
            "chapter.md",
            "林澈的发色不是银色。\n林澈的发色是银色。",
        )
        self.assertEqual(1, len(detect_issues(contradiction.directives)))

        compatible = parse_document(
            "compatible",
            "chapter.md",
            "林澈的发色不是银色。\n林澈的发色是黑色。",
        )
        self.assertEqual([], detect_issues(compatible.directives))

    def test_closed_fact_semantics_recognize_explicit_copular_negators(self):
        for index, negator in enumerate(("不是", "并非", "不为"), start=1):
            with self.subTest(negator=negator):
                row = ParsedDirective(
                    kind="fact",
                    attrs={"subject": "林澈", "predicate": "发色", "value": "银色"},
                    evidence=evidence(
                        f"林澈的发色{negator}银色。", line=index
                    ),
                )

                assessed, reason = assess_directive(row)

                self.assertIsNone(reason)
                self.assertIsNotNone(assessed)
                self.assertEqual("fact", assessed.kind)
                self.assertEqual("negated", assessed.attrs["modality"])
                self.assertEqual("negative", assessed.attrs["polarity"])
                self.assertTrue(eligible_for_deterministic_rules(assessed))

    def test_double_copular_negation_is_not_parsed_as_a_canonical_negative_fact(self):
        parsed = parse_document(
            "double-copular-negation",
            "chapter.md",
            "洛棠的档案访问级别并非不是临时观察员。",
        )

        canonical_facts = [
            row
            for row in parsed.directives
            if row.kind == "fact" and eligible_for_deterministic_rules(row)
        ]
        self.assertEqual([], canonical_facts)

    def test_closed_fact_semantics_recognize_bound_state_transition(self):
        row = ParsedDirective(
            kind="fact",
            attrs={
                "subject": "洛棠",
                "predicate": "档案访问级别",
                "value": "临时观察员",
            },
            evidence=evidence(
                "交接终端通过双人校验后亮起绿灯：洛棠的档案访问级别变成临时观察员；"
                "该条目属于终端强制结果，因而拒绝显示沉墨卷宗。"
            ),
        )

        assessed, reason = assess_directive(row)

        self.assertIsNone(reason)
        self.assertIsNotNone(assessed)
        self.assertEqual("fact", assessed.kind)
        self.assertEqual("asserted", assessed.attrs["modality"])
        self.assertEqual("affirmative", assessed.attrs["polarity"])
        self.assertTrue(eligible_for_deterministic_rules(assessed))

    def test_fact_relation_binds_complete_subject_and_passive_transition(self):
        attrs = {
            "subject": "洛棠",
            "predicate": "档案访问级别",
            "value": "临时观察员",
            "modality": "asserted",
            "source_scope": "narrator",
            "certainty": "certain",
        }
        passive, passive_reason = assess_directive(
            ParsedDirective(
                kind="fact",
                attrs=attrs,
                evidence=evidence(
                    "洛棠的档案访问级别已被管理员调整为临时观察员。"
                ),
                provenance_sources=frozenset({"model"}),
            )
        )
        substring, substring_reason = assess_directive(
            ParsedDirective(
                kind="fact",
                attrs=attrs,
                evidence=evidence(
                    "陆洛棠的档案访问级别已被管理员调整为临时观察员。",
                    line=2,
                ),
                provenance_sources=frozenset({"model"}),
            )
        )

        self.assertIsNone(passive_reason)
        self.assertIsNotNone(passive)
        self.assertTrue(eligible_for_deterministic_rules(passive))
        self.assertEqual("unbound_fact_relation", substring_reason)
        self.assertIsNotNone(substring)
        self.assertFalse(eligible_for_deterministic_rules(substring))

    def test_common_completed_fact_transition_forms_are_canonical(self):
        cases = (
            (
                {"subject": "洛棠", "predicate": "档案访问级别", "value": "临时观察员"},
                "管理员最终将洛棠的档案访问级别调整为临时观察员。",
            ),
            (
                {"subject": "洛棠", "predicate": "档案访问级别", "value": "临时观察员"},
                "洛棠的档案访问级别，现已调整为临时观察员。",
            ),
            (
                {"subject": "洛棠", "predicate": "档案访问级别", "value": "正式观察员"},
                "洛棠的档案访问级别不是临时观察员而是正式观察员。",
            ),
            (
                {"subject": "洛棠", "predicate": "档案访问级别", "value": "临时观察员"},
                "洛棠的档案访问级别变成临时观察员了。",
            ),
            (
                {"subject": "洛棠", "predicate": "档案访问级别", "value": "临时观察员"},
                "洛棠的档案访问级别是临时观察员的。",
            ),
        )
        for line, (fields, source) in enumerate(cases, start=1):
            with self.subTest(source=source):
                assessed, reason = assess_directive(
                    ParsedDirective(
                        kind="fact",
                        attrs={
                            **fields,
                            "modality": "asserted",
                            "source_scope": "narrator",
                            "certainty": "certain",
                        },
                        evidence=evidence(source, line=line),
                        provenance_sources=frozenset({"model"}),
                    )
                )

                self.assertIsNone(reason)
                self.assertIsNotNone(assessed)
                self.assertTrue(eligible_for_deterministic_rules(assessed))

    def test_contrast_sentence_also_binds_the_explicitly_negated_first_value(self):
        assessed, reason = assess_directive(
            ParsedDirective(
                kind="fact",
                attrs={
                    "subject": "洛棠",
                    "predicate": "档案访问级别",
                    "value": "临时观察员",
                    "modality": "asserted",
                    "source_scope": "narrator",
                    "certainty": "certain",
                },
                evidence=evidence(
                    "洛棠的档案访问级别不是临时观察员而是正式观察员。"
                ),
                provenance_sources=frozenset({"model"}),
            )
        )

        self.assertIsNone(reason)
        self.assertIsNotNone(assessed)
        self.assertEqual("negative", assessed.attrs["polarity"])
        self.assertTrue(eligible_for_deterministic_rules(assessed))

    def test_planned_or_unexecuted_fact_transition_is_never_canonical(self):
        sources = (
            "管理员计划将洛棠的档案访问级别调整为临时观察员。",
            "管理员计划把洛棠的档案访问级别设置为临时观察员。",
            "管理员要求将洛棠的档案访问级别设置成临时观察员。",
            "管理员并未将洛棠的档案访问级别调整为临时观察员。",
        )
        for line, source in enumerate(sources, start=1):
            with self.subTest(source=source):
                assessed, reason = assess_directive(
                    ParsedDirective(
                        kind="fact",
                        attrs={
                            "subject": "洛棠",
                            "predicate": "档案访问级别",
                            "value": "临时观察员",
                            "modality": "asserted",
                            "source_scope": "narrator",
                            "certainty": "certain",
                        },
                        evidence=evidence(source, line=line),
                        provenance_sources=frozenset({"model"}),
                    )
                )

                self.assertIn(reason, {"unrealized_fact_relation", "unbound_fact_relation"})
                self.assertIsNotNone(assessed)
                self.assertEqual("tentative_fact", assessed.kind)
                self.assertFalse(eligible_for_deterministic_rules(assessed))

    def test_fact_transition_requires_the_complete_submitted_value(self):
        assessed, reason = assess_directive(
            ParsedDirective(
                kind="fact",
                attrs={
                    "subject": "洛棠",
                    "predicate": "档案访问级别",
                    "value": "临时观察员",
                    "modality": "asserted",
                    "source_scope": "narrator",
                    "certainty": "certain",
                },
                evidence=evidence("洛棠的档案访问级别变成临时观察员助理。"),
                provenance_sources=frozenset({"model"}),
            )
        )

        self.assertEqual("unbound_fact_relation", reason)
        self.assertIsNotNone(assessed)
        self.assertEqual("tentative_fact", assessed.kind)
        self.assertFalse(eligible_for_deterministic_rules(assessed))

    def test_latest_authoritative_fact_support_wins_over_report_or_old_value(self):
        attrs = {
            "subject": "洛棠",
            "predicate": "档案访问级别",
            "value": "临时观察员",
        }
        confirmed, confirmed_reason = assess_directive(
            ParsedDirective(
                kind="fact",
                attrs=attrs,
                evidence=evidence(
                    "值班员转述：洛棠的档案访问级别是临时观察员。"
                    "随后，系统日志确认洛棠的档案访问级别是临时观察员。"
                ),
            )
        )
        corrected, corrected_reason = assess_directive(
            ParsedDirective(
                kind="fact",
                attrs=attrs,
                evidence=evidence(
                    "系统日志确认洛棠的档案访问级别是临时观察员。"
                    "随后系统日志修正：洛棠的档案访问级别不是临时观察员。"
                ),
            )
        )

        self.assertIsNone(confirmed_reason)
        self.assertIsNotNone(confirmed)
        self.assertEqual("asserted", confirmed.attrs["modality"])
        self.assertTrue(eligible_for_deterministic_rules(confirmed))
        self.assertIsNone(corrected_reason)
        self.assertIsNotNone(corrected)
        self.assertEqual("negated", corrected.attrs["modality"])
        self.assertEqual("negative", corrected.attrs["polarity"])

    def test_unrelated_copula_or_transition_does_not_close_fact_semantics(self):
        attrs = {
            "subject": "洛棠",
            "predicate": "档案访问级别",
            "value": "临时观察员",
        }
        sources = (
            "洛棠与临时观察员核对档案访问级别且该条目属于终端记录。",
            "洛棠与临时观察员核对档案访问级别时指示灯变成绿色。",
        )

        for line, source in enumerate(sources, start=1):
            with self.subTest(source=source):
                assessed, reason = assess_directive(
                    ParsedDirective(
                        kind="fact",
                        attrs=attrs,
                        evidence=evidence(source, line=line),
                    )
                )

                self.assertEqual("missing_semantic_labels", reason)
                self.assertIsNotNone(assessed)
                self.assertEqual("tentative_fact", assessed.kind)
                self.assertFalse(eligible_for_deterministic_rules(assessed))

    def test_dialogue_and_reported_document_have_different_source_scope(self):
        spoken = ParsedDirective(
            kind="fact",
            attrs={"subject": "林澈", "predicate": "身份", "value": "领航员"},
            evidence=evidence("苏弦说：“林澈的身份是领航员。”"),
        )
        archive = ParsedDirective(
            kind="fact",
            attrs={"subject": "林澈", "predicate": "身份", "value": "领航员"},
            evidence=evidence("档案记载：“林澈的身份是领航员。”", line=2),
        )
        unquoted_speech = ParsedDirective(
            kind="fact",
            attrs={"subject": "林澈", "predicate": "身份", "value": "领航员"},
            evidence=evidence("苏弦声称：林澈的身份是领航员。", line=3),
        )
        literal_quote = ParsedDirective(
            kind="fact",
            attrs={"subject": "林澈", "predicate": "身份", "value": "领航员"},
            evidence=evidence("纸页上只有一句：“林澈的身份是领航员。”", line=4),
        )
        quality = apply_semantic_quality_gate(
            [spoken, archive, unquoted_speech, literal_quote]
        )
        self.assertEqual("character_claim", quality.directives[0].kind)
        self.assertEqual("character_dialogue", quality.directives[0].attrs["source_scope"])
        self.assertEqual("fact", quality.directives[1].kind)
        self.assertEqual("quoted_material", quality.directives[1].attrs["source_scope"])
        self.assertEqual("character_claim", quality.directives[2].kind)
        self.assertEqual("quoted_material", quality.directives[3].attrs["source_scope"])
        self.assertFalse(eligible_for_deterministic_rules(quality.directives[3]))

    def test_full_evidence_attribution_blocks_semantic_laundering(self):
        cases = (
            (
                "旧卷只记载：佩戴青铜铃者可以听见潮声。",
                {"subject": "佩戴青铜铃者", "predicate": "能力", "value": "听见潮声"},
                "quoted_material",
            ),
            (
                "“洛岚盗走了星核。”值班员说道。",
                {"subject": "洛岚", "predicate": "盗走", "value": "星核"},
                "character_dialogue",
            ),
        )
        for text, core, expected_scope in cases:
            with self.subTest(text=text):
                assessed, _ = assess_directive(
                    ParsedDirective(
                        kind="fact",
                        attrs={
                            **core,
                            "modality": "asserted",
                            "source_scope": "narrator",
                            "certainty": "certain",
                        },
                        evidence=evidence(text),
                    )
                )
                self.assertIsNotNone(assessed)
                self.assertEqual(expected_scope, assessed.attrs["source_scope"])
                self.assertFalse(eligible_for_deterministic_rules(assessed))

    def test_reported_is_ineligible_except_for_observed_knowledge_claim(self):
        reported_fact = ParsedDirective(
            kind="fact",
            attrs={
                "subject": "洛岚",
                "predicate": "身份",
                "value": "领航员",
                "modality": "reported",
                "source_scope": "narrator",
                "certainty": "certain",
            },
            evidence=evidence("洛岚的身份是领航员。"),
        )
        observed_claim = ParsedDirective(
            kind="claims_knows",
            attrs={
                "character": "洛岚",
                "fact": "星门口令",
                "time": "1026-01-01 09:00",
                "modality": "reported",
                "source_scope": "character_dialogue",
                "certainty": "certain",
            },
            evidence=evidence("洛岚说出了星门口令。", line=2),
        )
        self.assertFalse(eligible_for_deterministic_rules(reported_fact))
        self.assertTrue(eligible_for_deterministic_rules(observed_claim))

    def test_semantic_gate_downgrades_command_plan_and_missing_execution(self):
        candidates = (
            (
                "总站长命令砚霜在寂灯环廊中启动余光信标。",
                "tentative_fact",
                "narrator",
            ),
            (
                "砚霜计划在寂灯环廊中启动余光信标。",
                "tentative_fact",
                "narrator",
            ),
            (
                (
                    "值班纪要转述：总站长命令砚霜在寂灯环廊中启动余光信标；"
                    "后续记录缺少任何执行结果。"
                ),
                "character_claim",
                "character_dialogue",
            ),
        )
        for source, expected_kind, expected_scope in candidates:
            with self.subTest(source=source):
                directive = ParsedDirective(
                    kind="world_assert",
                    attrs={
                        "key": "scope_action:寂灯环廊:余光信标",
                        "value": "performed",
                        "actor": "砚霜",
                        "modality": "asserted",
                        "source_scope": "narrator",
                        "certainty": "certain",
                    },
                    evidence=evidence(source),
                    provenance_sources=frozenset({"model"}),
                )
                assessed, reason = assess_directive(directive)
                self.assertEqual("unrealized_action", reason)
                self.assertIsNotNone(assessed)
                self.assertEqual(expected_kind, assessed.kind)
                self.assertEqual(expected_scope, assessed.attrs["source_scope"])
                self.assertFalse(eligible_for_deterministic_rules(assessed))

    def test_semantic_gate_keeps_plain_performed_action_asserted(self):
        directive = ParsedDirective(
            kind="world_assert",
            attrs={
                "key": "scope_action:寂灯环廊:余光信标",
                "value": "performed",
                "actor": "砚霜",
                "modality": "asserted",
                "source_scope": "narrator",
                "certainty": "certain",
            },
            evidence=evidence("砚霜在寂灯环廊中启动余光信标，指示灯当场亮起。"),
        )
        assessed, reason = assess_directive(directive)
        self.assertIsNone(reason)
        self.assertEqual("world_assert", assessed.kind)
        self.assertTrue(eligible_for_deterministic_rules(assessed))

    def test_closed_use_semantics_recognize_bound_actual_operations(self):
        rows = (
            (
                "尹夙",
                "回声纺锤",
                "2027-03-09 09:15，尹夙亲自操作了回声纺锤。",
            ),
            ("尹夙", "回声纺锤", "尹夙操控回声纺锤。"),
            ("尹夙", "回声纺锤", "尹夙按下回声纺锤。"),
            ("尹夙", "回声纺锤", "尹夙将回声纺锤插入基座。"),
            ("尹夙", "回声纺锤", "尹夙在修复室操作了回声纺锤。"),
            ("尹夙", "回声纺锤", "尹夙取得授权后实际使用了回声纺锤。"),
            ("尹夙", "回声纺锤", "回声纺锤被尹夙启用。"),
            ("尹夙", "回声纺锤", "回声纺锤已被尹夙启用。"),
            ("尹夙", "回声纺锤", "回声纺锤已被尹夙亲手启用。"),
            ("尹夙", "回声纺锤", "尹夙对回声纺锤进行了操作。"),
        )
        for line, (user, item, source) in enumerate(rows, start=1):
            with self.subTest(source=source):
                assessed, reason = assess_directive(
                    ParsedDirective(
                        kind="uses",
                        attrs={"user": user, "item": item},
                        evidence=evidence(source, line=line),
                    )
                )

                self.assertIsNone(reason)
                self.assertIsNotNone(assessed)
                self.assertEqual("uses", assessed.kind)
                self.assertEqual("asserted", assessed.attrs["modality"])
                self.assertTrue(eligible_for_deterministic_rules(assessed))

    def test_use_semantics_reject_non_actions_and_unrealized_or_reported_uses(self):
        rows = (
            (
                "尹夙阅读回声纺锤的操作说明。",
                "tentative_fact",
                "missing_semantic_labels",
            ),
            (
                "尹夙拥有操作回声纺锤的权限。",
                "tentative_fact",
                "missing_semantic_labels",
            ),
            ("总站长命令尹夙操作回声纺锤。", "tentative_fact", "unrealized_action"),
            ("尹夙计划操作回声纺锤。", "tentative_fact", "unrealized_action"),
            ("尹夙尚未操作回声纺锤。", "tentative_fact", "unrealized_action"),
            ("尹夙并未将回声纺锤启动。", "tentative_fact", "unrealized_action"),
            ("尹夙尚未把回声纺锤拿出。", "tentative_fact", "unrealized_action"),
            (
                "值班纪要转述：尹夙操作了回声纺锤。",
                "character_claim",
                "non_authoritative_source",
            ),
        )
        for line, (source, expected_kind, expected_reason) in enumerate(rows, start=1):
            with self.subTest(source=source):
                assessed, reason = assess_directive(
                    ParsedDirective(
                        kind="uses",
                        attrs={"user": "尹夙", "item": "回声纺锤"},
                        evidence=evidence(source, line=line),
                    )
                )

                self.assertEqual(expected_reason, reason)
                self.assertIsNotNone(assessed)
                self.assertEqual(expected_kind, assessed.kind)
                self.assertFalse(eligible_for_deterministic_rules(assessed))

    def test_declared_model_labels_cannot_upgrade_an_unbound_use_relation(self):
        for line, source in enumerate(
            (
                "尹夙阅读回声纺锤的操作说明。",
                "尹夙拥有操作回声纺锤的权限。",
                "陆尹夙亲自操作了回声纺锤。",
                "尹夙亲自操作了回声纺锤模型。",
                "尹夙亲自操作了回声纺锤外壳。",
                "尹夙亲自操作了回声纺锤启动器。",
                "尹夙亲自操作了回声纺锤打开器。",
                "尹夙亲自使用了回声纺锤后盖。",
                "尹夙亲自使用了回声纺锤时钟。",
                "尹夙亲自使用了回声纺锤的功能模块。",
                "尹夙亲自使用了回声纺锤并联器。",
                "尹夙亲自使用了回声纺锤启动器后离开。",
                "尹夙亲自使用了回声纺锤打开器时停电。",
            ),
            start=1,
        ):
            with self.subTest(source=source):
                assessed, reason = assess_directive(
                    ParsedDirective(
                        kind="uses",
                        attrs={
                            "user": "尹夙",
                            "item": "回声纺锤",
                            "modality": "asserted",
                            "source_scope": "narrator",
                            "certainty": "certain",
                        },
                        evidence=evidence(source, line=line),
                        provenance_sources=frozenset({"model"}),
                    )
                )

                self.assertEqual("unbound_use_relation", reason)
                self.assertIsNotNone(assessed)
                self.assertEqual("tentative_fact", assessed.kind)
                self.assertFalse(eligible_for_deterministic_rules(assessed))

    def test_colon_framed_plan_permission_and_rehearsal_are_not_execution(self):
        for line, source in enumerate(
            (
                "操作计划：尹夙使用回声纺锤。",
                "核验计划：尹夙使用回声纺锤。",
                "撤离方案：尹夙操作回声纺锤。",
                "安全规范：尹夙使用回声纺锤。",
                "设备使用规范：尹夙操作回声纺锤。",
                "战斗预案：尹夙使用回声纺锤。",
                "巡检流程：尹夙操作回声纺锤。",
                "编辑安排：尹夙使用回声纺锤。",
                "舰内设备核验要求：尹夙操作回声纺锤。",
                "计划如下：尹夙使用回声纺锤。",
                "长线演练执行方案如下所示：尹夙操作回声纺锤。",
            ),
            start=1,
        ):
            with self.subTest(source=source):
                assessed, reason = assess_directive(
                    ParsedDirective(
                        kind="uses",
                        attrs={
                            "user": "尹夙",
                            "item": "回声纺锤",
                            "modality": "asserted",
                            "source_scope": "narrator",
                            "certainty": "certain",
                        },
                        evidence=evidence(source, line=line),
                        provenance_sources=frozenset({"model"}),
                    )
                )

                self.assertEqual("unrealized_action", reason)
                self.assertIsNotNone(assessed)
                self.assertFalse(eligible_for_deterministic_rules(assessed))

    def test_generic_colon_headings_do_not_assert_fact_relations(self):
        headings = (
            "核验计划",
            "撤离方案",
            "安全规范",
            "设备使用规范",
            "战斗预案",
            "巡检流程",
            "编辑安排",
        )
        for line, heading in enumerate(headings, start=1):
            with self.subTest(heading=heading):
                assessed, reason = assess_directive(
                    ParsedDirective(
                        kind="fact",
                        attrs={
                            "subject": "洛棠",
                            "predicate": "档案访问级别",
                            "value": "临时观察员",
                            "modality": "asserted",
                            "source_scope": "narrator",
                            "certainty": "certain",
                        },
                        evidence=evidence(
                            f"{heading}：洛棠的档案访问级别设为临时观察员。",
                            line=line,
                        ),
                        provenance_sources=frozenset({"model"}),
                    )
                )

                self.assertEqual("unrealized_fact_relation", reason)
                self.assertIsNotNone(assessed)
                self.assertEqual("tentative_fact", assessed.kind)
                self.assertFalse(eligible_for_deterministic_rules(assessed))

    def test_completed_plan_bridge_still_allows_later_actual_use(self):
        assessed, reason = assess_directive(
            ParsedDirective(
                kind="uses",
                attrs={
                    "user": "尹夙",
                    "item": "回声纺锤",
                    "modality": "asserted",
                    "source_scope": "narrator",
                    "certainty": "certain",
                },
                evidence=evidence(
                    "计划完成后，尹夙实际使用了回声纺锤。"
                ),
                provenance_sources=frozenset({"model"}),
            )
        )

        self.assertIsNone(reason)
        self.assertIsNotNone(assessed)
        self.assertTrue(eligible_for_deterministic_rules(assessed))

    def test_instructional_or_permission_use_is_not_promoted(self):
        sources = (
            "用户指南：尹夙使用回声纺锤时应先断电。",
            "维护规程：尹夙操作回声纺锤时需要双人监护。",
            "系统允许尹夙操作回声纺锤。",
            "管理员授权尹夙使用回声纺锤。",
        )
        for line, source in enumerate(sources, start=1):
            with self.subTest(source=source):
                assessed, _ = assess_directive(
                    ParsedDirective(
                        kind="uses",
                        attrs={
                            "user": "尹夙",
                            "item": "回声纺锤",
                            "modality": "asserted",
                            "source_scope": "narrator",
                            "certainty": "certain",
                        },
                        evidence=evidence(source, line=line),
                        provenance_sources=frozenset({"model"}),
                    )
                )

                self.assertIsNotNone(assessed)
                self.assertFalse(eligible_for_deterministic_rules(assessed))

    def test_latest_authoritative_use_support_wins_over_prior_report(self):
        assessed, reason = assess_directive(
            ParsedDirective(
                kind="uses",
                attrs={"user": "尹夙", "item": "回声纺锤"},
                evidence=evidence(
                    "值班员转述：尹夙操作了回声纺锤。"
                    "随后，尹夙亲自操作了回声纺锤。"
                ),
            )
        )

        self.assertIsNone(reason)
        self.assertIsNotNone(assessed)
        self.assertEqual("uses", assessed.kind)
        self.assertEqual("narrator", assessed.attrs["source_scope"])
        self.assertTrue(eligible_for_deterministic_rules(assessed))

    def test_semantic_gate_localizes_action_modality_and_source(self):
        sources = (
            (
                "现场没有任何执行日志，但砚霜确实在寂灯环廊中启动余光信标，"
                "指示灯当场亮起。"
            ),
            (
                "总站长命令砚霜在寂灯环廊中启动余光信标。"
                "随后砚霜在寂灯环廊中启动余光信标，指示灯当场亮起。"
            ),
            (
                "值班员转述：昨夜风暴猛烈。"
                "随后砚霜在寂灯环廊中启动余光信标，指示灯当场亮起。"
            ),
            (
                "未经核验的报告声称旧桥已经坍塌。"
                "随后砚霜在寂灯环廊中启动余光信标，指示灯当场亮起。"
            ),
        )
        for source in sources:
            with self.subTest(source=source):
                directive = ParsedDirective(
                    kind="world_assert",
                    attrs={
                        "key": "scope_action:寂灯环廊:余光信标",
                        "value": "performed",
                        "actor": "砚霜",
                        "modality": "asserted",
                        "source_scope": "narrator",
                        "certainty": "certain",
                    },
                    evidence=evidence(source),
                    provenance_sources=frozenset({"model"}),
                )
                assessed, reason = assess_directive(directive)
                self.assertIsNone(reason)
                self.assertEqual("world_assert", assessed.kind)
                self.assertEqual("narrator", assessed.attrs["source_scope"])
                self.assertTrue(eligible_for_deterministic_rules(assessed))

    def test_explicit_performed_directive_is_not_downgraded_by_missing_telemetry(self):
        parsed = parse_document(
            "explicit-action",
            "canon.md",
            (
                '@world_assert key="scope_action:寂灯环廊:余光信标" '
                'value=performed actor=砚霜 | '
                "现场没有任何执行日志，但砚霜确实在寂灯环廊中启动余光信标，"
                "指示灯当场亮起。"
            ),
        )
        self.assertEqual(1, len(parsed.directives))
        assessed, reason = assess_directive(parsed.directives[0])
        self.assertIsNone(reason)
        self.assertEqual("world_assert", assessed.kind)
        self.assertEqual("directive", assessed.attrs["input_form"])
        self.assertTrue(eligible_for_deterministic_rules(assessed))

    def test_noncanonical_frame_is_internal_and_does_not_change_fingerprint(self):
        row = ParsedDirective(
            kind="event",
            attrs={
                "time": "1026-01-01 10:00",
                "location": "雪原",
                "participants": "沈砚",
                "modality": "asserted",
                "source_scope": "narrator",
                "certainty": "certain",
            },
            evidence=evidence("1026-01-01 10:00，沈砚在雪原。"),
        )
        framed = row.model_copy(update={"noncanonical_frame": True})
        self.assertNotIn("noncanonical_frame", framed.model_dump())
        self.assertEqual(row.model_dump(), framed.model_dump())
        self.assertEqual(
            directive_fingerprint(
                row, path="chapter.md", semantic_class="confirmed_narrative", eligible=True
            ),
            directive_fingerprint(
                framed,
                path="chapter.md",
                semantic_class="confirmed_narrative",
                eligible=True,
            ),
        )

    def test_dream_context_is_traceable_but_cannot_create_location_conflict(self):
        from app.candidate_normalizer import NormalizationResult

        class EventExtractor:
            def extract(self, document):
                line = 2 if document.id == "dream" else 1
                source = document.content.splitlines()[line - 1]
                location = "雪原" if document.id == "dream" else "潮痕镇"
                return ParsedDocument(
                    document_id=document.id,
                    document_name=document.name,
                    directives=[
                        ParsedDirective(
                            kind="event",
                            attrs={
                                "time": "1026-01-01 10:00",
                                "location": location,
                                "participants": "沈砚",
                                "modality": "asserted",
                                "source_scope": "narrator",
                                "certainty": "certain",
                            },
                            evidence=EvidenceSpan(
                                document_id=document.id,
                                document_name=document.name,
                                line_start=line,
                                line_end=line,
                                text=source,
                            ),
                        )
                    ],
                )

        class PassthroughNormalizer:
            def enrich(self, _documents, directives):
                return NormalizationResult(directives=list(directives))

        documents = [
            DocumentInput(
                "dream",
                "dream.md",
                "作者旁注：以下段落均为梦境。\n1026-01-01 10:00，沈砚站在雪原。",
            ),
            DocumentInput(
                "reality", "reality.md", "1026-01-01 10:00，沈砚站在潮痕镇。"
            ),
        ]
        result = AnalysisPipeline(
            extractor=EventExtractor(), normalizer=PassthroughNormalizer()
        ).run(documents)
        dream_rows = [
            row for row in result.directives if row.evidence.document_id == "dream"
        ]
        self.assertEqual(1, len(dream_rows))
        self.assertEqual("tentative_fact", dream_rows[0].kind)
        self.assertTrue(dream_rows[0].noncanonical_frame)
        self.assertFalse(eligible_for_deterministic_rules(dream_rows[0]))
        self.assertEqual([], result.issues)

    def test_reality_resumption_and_explicit_negation_are_not_overblocked(self):
        content = (
            "作者旁注：以下段落均为梦境。\n"
            "从梦中醒来后在现实，1026-01-01 10:00，沈砚站在潮痕镇；"
            "旁白明确说明，这不是回忆、幻象或通讯投影。"
        )
        self.assertFalse(document_context_has_noncanonical_frame(content, 2, 2))
        self.assertTrue(
            document_context_has_noncanonical_frame(
                "这不是回忆，而是梦境中的片段。", 1, 1
            )
        )

    def test_unverified_report_and_possible_certainty_never_become_canonical(self):
        anonymous = ParsedDirective(
            kind="fact",
            attrs={"subject": "账册", "predicate": "状态", "value": "已烧毁"},
            evidence=evidence("匿名信中声称：蓝色账册已经烧毁，真伪未验证。"),
        )
        possible = ParsedDirective(
            kind="fact",
            attrs={
                "subject": "陆遥",
                "predicate": "路线",
                "value": "暗渠",
                "certainty": "possible",
            },
            evidence=evidence("陆遥从暗渠抵达井底。", line=2),
        )
        quality = apply_semantic_quality_gate([anonymous, possible])
        self.assertEqual("character_claim", quality.directives[0].kind)
        self.assertEqual("unverified_report", quality.directives[0].attrs["source_scope"])
        self.assertEqual("tentative_fact", quality.directives[1].kind)
        self.assertEqual([], detect_issues(quality.directives))

    def test_rhetorical_question_is_traceable_but_never_a_fact(self):
        parsed = parse_document(
            "rhetorical", "chapter.md", "难道列车员会出售有毒食物吗？"
        )
        self.assertEqual(1, len(parsed.directives))
        self.assertEqual("open_question", parsed.directives[0].kind)
        self.assertEqual("rhetorical", parsed.directives[0].attrs["question_type"])
        self.assertEqual([], detect_issues(parsed.directives))

        dialogue = parse_document(
            "dialogue-question",
            "chapter.md",
            "她问同伴：“列车员推车上的食物真的安全吗？”同伴没有回答。",
        )
        self.assertEqual(
            "character_dialogue", dialogue.directives[0].attrs["source_scope"]
        )

    def test_ordinary_asserted_fact_keeps_existing_rule_behavior(self):
        parsed = parse_document(
            "ordinary",
            "chapter.md",
            "林澈的身份是领航员。\n林澈的身份是档案官。",
        )
        self.assertEqual(2, len([row for row in parsed.directives if row.kind == "fact"]))
        self.assertEqual(1, len(detect_issues(parsed.directives)))

    def test_branch_scope_is_formal_input_and_sibling_branches_do_not_conflict(self):
        route_a = DocumentInput(
            "a",
            "same-name.md",
            "1026-04-03 10:00，弥音在观星厅。",
            role="story_branch",
            scope="route_a",
        )
        route_b = DocumentInput(
            "b",
            "same-name.md",
            "1026-04-03 10:00，弥音在下层货舱。",
            role="story_branch",
            scope="route_b",
        )
        separated = AnalysisPipeline(extractor=BaselineExtractor()).run([route_a, route_b])
        self.assertEqual([], separated.issues)
        self.assertEqual(
            {"route_a", "route_b"},
            {row.attrs["story_scope"] for row in separated.directives if row.kind == "event"},
        )

        same_branch = AnalysisPipeline(extractor=BaselineExtractor()).run(
            [
                route_a,
                DocumentInput(
                    "a2",
                    "another-name.md",
                    "1026-04-03 10:00，弥音在下层货舱。",
                    role="story_branch",
                    scope="route_a",
                ),
            ]
        )
        self.assertEqual(1, len(same_branch.issues))

    def test_global_scope_meets_each_branch_but_sibling_branches_stay_isolated(self):
        global_fact = DocumentInput(
            "global",
            "canon.md",
            "角色甲的身份是领航员。",
            role="canon",
            scope="global",
        )
        branch_a = DocumentInput(
            "branch-a", "a.md", "角色甲的身份是档案官。", scope="branch_a"
        )
        branch_b = DocumentInput(
            "branch-b", "b.md", "角色甲的身份是领航员。", scope="branch_b"
        )
        result = AnalysisPipeline(extractor=BaselineExtractor()).run(
            [global_fact, branch_a, branch_b]
        )
        conflicts = [row for row in result.issues if row.category.value == "fact_conflict"]
        self.assertEqual(1, len(conflicts))
        self.assertEqual(
            {"canon.md", "a.md"}, {span.document_name for span in conflicts[0].evidence}
        )

    def test_matching_travel_limit_is_included_in_location_conflict_evidence(self):
        parsed = parse_document(
            "travel",
            "chapter.md",
            "普通人在十分钟内不可能往返雾港与山门。\n"
            "1028-01-02 09:00，角色甲在雾港。\n"
            "1028-01-02 09:00，角色甲在山门。",
        )
        issue = next(
            row for row in detect_issues(parsed.directives)
            if row.category.value == "location_collision"
        )
        self.assertEqual([1, 2, 3], [span.line_start for span in issue.evidence])

        hedged = parse_document(
            "hedged",
            "chapter.md",
            "普通人在十分钟内未必不可能往返雾港与山门。",
        )
        self.assertFalse(
            any(row.attrs.get("predicate") == "mobility_limit" for row in hedged.directives)
        )

    def test_explicit_denied_authorization_is_part_of_rule_conflict_evidence(self):
        rows = [
            ParsedDirective(
                kind="world_rule",
                attrs={
                    "key": "scope_action:封锁层:进入",
                    "value": "disabled",
                    "story_scope": "global",
                    "modality": "asserted",
                    "source_scope": "world_rule",
                    "certainty": "certain",
                },
                evidence=evidence("封锁层不得进入，除非获得书面授权。", line=1),
            ),
            ParsedDirective(
                kind="fact",
                attrs={
                    "subject": "角色甲",
                    "predicate": "authorization",
                    "value": "书面授权",
                    "polarity": "negative",
                    "modality": "negated",
                    "source_scope": "narrator",
                    "certainty": "certain",
                    "story_scope": "branch_a",
                },
                evidence=evidence("旁白确认角色甲没有取得书面授权。", line=2),
            ),
            ParsedDirective(
                kind="world_assert",
                attrs={
                    "key": "scope_action:封锁层:进入",
                    "value": "performed",
                    "actor": "角色甲",
                    "story_scope": "branch_a",
                    "modality": "asserted",
                    "source_scope": "narrator",
                    "certainty": "certain",
                },
                evidence=evidence("角色甲进入封锁层。", line=3),
            ),
        ]
        issue = next(
            row for row in detect_issues(rows)
            if row.category.value == "world_rule_conflict"
        )
        self.assertEqual([1, 2, 3], [span.line_start for span in issue.evidence])

    def test_explicit_name_title_binding_normalizes_but_cooccurrence_does_not(self):
        bound = AnalysisPipeline(extractor=BaselineExtractor()).run(
            [
                DocumentInput(
                    "bound",
                    "bound.md",
                    "1028-01-02 09:00，船长顾青在雾港。\n"
                    "1028-01-02 09:00，顾青在山门。",
                )
            ]
        )
        self.assertTrue(
            any(row.category.value == "location_collision" for row in bound.issues)
        )
        self.assertTrue(
            all(
                row.attrs["participants"] == "顾青"
                for row in bound.directives
                if row.kind == "event"
            )
        )

        unbound = AnalysisPipeline(extractor=BaselineExtractor()).run(
            [
                DocumentInput(
                    "unbound",
                    "unbound.md",
                    "1028-01-02 09:00，顾青与老船长在雾港。\n"
                    "1028-01-02 09:00，老船长在山门。",
                )
            ]
        )
        self.assertFalse(
            any(row.category.value == "location_collision" for row in unbound.issues)
        )

    def test_clarification_can_coexist_with_an_independent_hard_conflict(self):
        parsed = parse_document(
            "coexist",
            "chapter.md",
            "林澈的身份是领航员。\n林澈的身份是档案官。",
        )
        clarification = ParsedDirective(
            kind="clarification",
            attrs={
                "summary": "推车食物是否属于餐车供应范围需要确认",
                "category": "scope_unknown",
            },
            evidence=evidence("推车食物的供应范围尚未说明。", line=3),
        )
        quality = apply_semantic_quality_gate([*parsed.directives, clarification])
        self.assertEqual(1, len(detect_issues(quality.directives)))
        self.assertEqual(1, len([row for row in quality.directives if row.kind == "clarification"]))

    def test_clarification_with_conflict_shaped_fields_never_enters_rules(self):
        clarification = ParsedDirective(
            kind="clarification",
            attrs={
                "summary": "两处身份描述需要确认是否属于同一阶段",
                "category": "missing_state_transition",
                "subject": "角色甲",
                "predicate": "身份",
                "value": "领航员",
            },
            evidence=evidence("角色甲的阶段尚未说明，需要确认。"),
        )
        quality = apply_semantic_quality_gate([clarification])
        self.assertEqual("clarification", quality.directives[0].kind)
        self.assertEqual([], detect_issues(quality.directives))

    def test_impossibility_is_definite_but_double_negation_is_not(self):
        definite = ParsedDirective(
            kind="fact",
            attrs={"subject": "角色甲", "predicate": "抵达", "value": "山门"},
            evidence=evidence("角色甲不可能抵达山门。"),
        )
        double_negative = definite.model_copy(
            update={"evidence": evidence("角色甲不可能不抵达山门。", line=2)}
        )
        hedged = definite.model_copy(
            update={"evidence": evidence("角色甲未必不可能抵达山门。", line=3)}
        )
        result = apply_semantic_quality_gate([definite, double_negative, hedged])
        self.assertEqual("fact", result.directives[0].kind)
        self.assertEqual("negated", result.directives[0].attrs["modality"])
        self.assertEqual("negative", result.directives[0].attrs["polarity"])
        self.assertTrue(all(row.kind == "tentative_fact" for row in result.directives[1:]))

    def test_quality_gate_is_idempotent_for_noncanonical_records(self):
        first = parse_document(
            "idempotent", "chapter.md", "难道列车员会出售有毒食物吗？"
        ).directives
        second = apply_semantic_quality_gate(first)
        self.assertEqual(first, second.directives)
        self.assertEqual(0, second.transformed_count)


class ModelSemanticQualityTests(unittest.TestCase):
    def provider_for(self, records: list[dict]) -> OpenAICompatibleProvider:
        payload = json.dumps({"records": records}, ensure_ascii=False)
        return OpenAICompatibleProvider(
            settings(),
            transport=httpx.MockTransport(lambda _: completion(payload)),
        )

    def test_model_cannot_upgrade_question_to_asserted_fact(self):
        text = "那么他推车上售卖的食物是否等同于餐车提供的食物？"
        record = {
            "kind": "fact",
            "subject": "那么他推车上售卖",
            "predicate": "食物",
            "value": "否等同于餐车提供的食物",
            "modality": "asserted",
            "source_scope": "narrator",
            "certainty": "certain",
            "source_line_start": 1,
            "source_line_end": 1,
        }
        result = ModelEnhancedExtractor(self.provider_for([record])).extract(
            DocumentInput("model-question", "chapter.md", text)
        )
        self.assertFalse(any(row.kind == "fact" for row in result.directives))
        self.assertEqual(1, len([row for row in result.directives if row.kind == "open_question"]))
        self.assertEqual([], detect_issues(result.directives))

    def test_successful_model_pass_cannot_preserve_unsafe_baseline_fact(self):
        text = "那么他推车上售卖的食物是否等同于餐车提供的食物？"

        class UnsafeBaseline:
            def extract(self, document):
                return ParsedDocument(
                    document_id=document.id,
                    document_name=document.name,
                    directives=[
                        ParsedDirective(
                            kind="fact",
                            attrs={
                                "subject": "那么他推车上售卖",
                                "predicate": "食物",
                                "value": "否等同于餐车提供的食物",
                            },
                            evidence=EvidenceSpan(
                                document_id=document.id,
                                document_name=document.name,
                                line_start=1,
                                line_end=1,
                                text=text,
                            ),
                        )
                    ],
                )

        result = ModelEnhancedExtractor(
            self.provider_for([]), baseline=UnsafeBaseline()
        ).extract(DocumentInput("unsafe-baseline", "chapter.md", text))
        # A schema-valid empty envelope is transport success, but it is not
        # evidence that the model semantically covered this non-empty chunk.
        self.assertFalse(result.model_used)
        self.assertEqual(1, result.model_execution.empty_response_chunks)
        self.assertTrue(any("无法证明完整覆盖" in row for row in result.warnings))
        self.assertFalse(any(row.kind == "fact" for row in result.directives))
        self.assertEqual(1, len([row for row in result.directives if row.kind == "open_question"]))

    def test_model_semantic_enums_are_strongly_validated(self):
        record = {
            "kind": "fact",
            "subject": "林澈",
            "predicate": "身份",
            "value": "领航员",
            "modality": "absolutely_true",
            "source_scope": "narrator",
            "certainty": "certain",
            "source_line_start": 1,
            "source_line_end": 1,
        }
        result = ModelEnhancedExtractor(self.provider_for([record])).extract(
            DocumentInput("invalid-enum", "chapter.md", "林澈的身份是领航员。")
        )
        self.assertFalse(result.model_used)
        self.assertEqual(1, len([row for row in result.directives if row.kind == "fact"]))
        self.assertTrue(any("schema_validation" in warning for warning in result.warnings))

    def test_missing_model_semantic_labels_are_rejected_not_defaulted(self):
        with self.assertRaises(ValidationError):
            RECORD_ADAPTER.validate_python(
                {
                    "kind": "fact",
                    "subject": "角色甲",
                    "predicate": "身份",
                    "value": "守卫",
                    "source_line_start": 1,
                    "source_line_end": 1,
                }
            )

    def test_pipeline_filters_question_candidate_added_after_extraction(self):
        class UnsafeNormalizer:
            def enrich(self, _documents, directives):
                from app.candidate_normalizer import NormalizationResult

                unsafe = ParsedDirective(
                    kind="fact",
                    attrs={
                        "subject": "那么他推车上售卖",
                        "predicate": "食物",
                        "value": "否等同于餐车提供的食物",
                    },
                    evidence=evidence("那么他推车上售卖的食物是否等同于餐车提供的食物？"),
                )
                return NormalizationResult(directives=[*directives, unsafe])

        result = AnalysisPipeline(
            extractor=ModelEnhancedExtractor(self.provider_for([])),
            normalizer=UnsafeNormalizer(),
        ).run([DocumentInput("pipeline", "chapter.md", "普通背景。")])
        self.assertFalse(any(row.kind == "fact" for row in result.directives))
        self.assertEqual(1, len([row for row in result.directives if row.kind == "open_question"]))

    def test_cancellation_before_second_chunk_does_not_fail_or_open_circuit(self):
        calls = 0

        def handler(_request):
            nonlocal calls
            calls += 1
            return completion('{"records":[]}')

        provider = OpenAICompatibleProvider(
            settings(model_chunk_max_chars=32, model_chunk_overlap_lines=0),
            transport=httpx.MockTransport(handler),
        )
        extractor = ModelEnhancedExtractor(provider)
        checkpoints = 0

        def checkpoint():
            nonlocal checkpoints
            checkpoints += 1
            # document-before, chunk-1-before, chunk-1-after, chunk-2-before
            if checkpoints == 4:
                raise AnalysisCancelled("cancelled by test")

        with self.assertRaises(AnalysisCancelled):
            AnalysisPipeline(extractor=extractor).run(
                [
                    DocumentInput(
                        "cancel",
                        "chapter.md",
                        "第一段普通背景信息足够触发一个模型分块。\n"
                        "第二段普通背景信息足够触发第二个模型分块。\n"
                        "第三段普通背景信息继续扩充分块数量。",
                    )
                ],
                checkpoint=checkpoint,
            )
        self.assertEqual(1, calls)
        self.assertFalse(extractor._circuit_open)
        self.assertEqual(0, extractor._failed_documents)


if __name__ == "__main__":
    unittest.main()
