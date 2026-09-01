from __future__ import annotations

from collections import defaultdict

from .domain import EvaluationResult, IssueCategory
from .parser import parse_document
from .rules import detect_issues


CATEGORIES = list(IssueCategory)


def generate_cases(per_category: int = 16) -> list[dict]:
    """Generate explicit directives for rule-engine regression, not NLP evaluation."""
    cases: list[dict] = []
    for i in range(per_category):
        name = f"角色{i:02d}"
        cases.extend([
            {"id": f"fact-{i}", "expected": "fact_conflict", "text": f'@fact subject="{name}" predicate="发色" value="银色" | 初始设定。\n@fact subject="{name}" predicate="发色" value="黑色" | 后续章节。'},
            {"id": f"location-{i}", "expected": "location_collision", "text": f'@event id="a{i}" time="1026-01-01T10:00" location="北港" participants="{name}" | 位于北港。\n@event id="b{i}" time="1026-01-01T10:00" location="南塔" participants="{name}" | 位于南塔。'},
            {"id": f"knowledge-{i}", "expected": "knowledge_without_acquisition", "text": f'@claims_knows character="{name}" fact="门禁密码" time="1026-01-01T09:00" | 提前说出密码。\n@knows character="{name}" fact="门禁密码" time="1026-01-01T12:00" | 中午才获知。'},
            {"id": f"item-{i}", "expected": "item_ownership", "text": f'@item item="钥匙{i}" owner="苏弦" time="1026-01-01T08:00" | 苏弦保管钥匙。\n@uses item="钥匙{i}" user="{name}" time="1026-01-01T09:00" | 角色使用钥匙。'},
            {"id": f"rule-{i}", "expected": "world_rule_conflict", "text": f'@world_rule key="星门能源{i}" value="潮汐晶核" | 权威规则。\n@world_assert key="星门能源{i}" value="普通火焰" | 章节中的冲突描述。'},
        ])
    return cases


def run_evaluation() -> EvaluationResult:
    """Run the deterministic @directive wiring regression."""
    cases = generate_cases()
    tp = fp = fn = evidence_hits = 0
    by_category = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for case in cases:
        parsed = parse_document(case["id"], f'{case["id"]}.md', case["text"])
        detected = {issue.category.value: issue for issue in detect_issues(parsed.directives)}
        expected = case["expected"]
        if expected in detected:
            tp += 1
            by_category[expected]["tp"] += 1
            if len(detected[expected].evidence) >= 1:
                evidence_hits += 1
        else:
            fn += 1
            by_category[expected]["fn"] += 1
        for extra in set(detected) - {expected}:
            fp += 1
            by_category[extra]["fp"] += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    category_scores = {}
    for category in CATEGORIES:
        row = by_category[category.value]
        p = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0.0
        r = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0.0
        category_scores[category.value] = {**row, "precision": p, "recall": r}
    return EvaluationResult(
        sample_count=len(cases),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        evidence_hit_rate=evidence_hits / len(cases),
        category_scores=category_scores,
    )
