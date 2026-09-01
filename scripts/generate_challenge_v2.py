from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


SEED = 20260902
NAMES = ("叶峤", "温岚", "谢舟", "白榆", "闻溪", "楚遥", "许澄", "顾野", "程霁", "洛川")
PLACES = (("雾港", "山门"), ("石桥", "云台"), ("北仓", "灯塔"), ("河湾", "钟楼"), ("旧站", "星庭"))


def issue(category: str, *evidence: tuple[str, int]) -> list[dict]:
    return [{"category": category, "evidence": [{"document": name, "line": line} for name, line in evidence]}]


def case(category: str, index: int, polarity: str, documents: list[dict], expected: list[dict], pattern: str) -> dict:
    case_id = f"challenge-v2-{category}-{polarity}-{index:02d}"
    return {
        "case_id": case_id,
        "scenario_id": case_id,
        "split": "test",
        "category_focus": category,
        "polarity": polarity,
        "documents": documents,
        "expected_issues": expected,
        "generation": {
            "kind": "fixed-template synthetic natural-language",
            "seed": SEED,
            "pattern": pattern,
            "developer_visible": True,
        },
    }


def build() -> list[dict]:
    random.Random(SEED)  # Seed is recorded even though template order is deliberately fixed.
    rows: list[dict] = []

    for i in range(5):
        name = NAMES[i]
        rows.append(case("fact_conflict", i, "positive", [
            {"name": "world.md", "content": f"{name}的瞳孔颜色是琥珀色。"},
            {"name": "chapter.md", "content": f"{name}的瞳孔颜色为灰蓝色。"},
        ], issue("fact_conflict", ("world.md", 1), ("chapter.md", 1)), "stable attribute contradiction"))
        rows.append(case("fact_conflict", i, "hard-negative", [
            {"name": "world.md", "content": f"{name}的制服颜色是墨绿色。"},
            {"name": "chapter.md", "content": f"{NAMES[i + 5]}的制服颜色是赭红色。"},
        ], [], "same predicate but different subjects"))

    for i in range(5):
        name = NAMES[i]
        start, end = PLACES[i]
        stamp = f"1027-03-{i + 1:02d} 09:00"
        positive_permits = (
            f"管理局只向{NAMES[i + 5]}签发了从{start}到{end}的瞬时通行许可，有效期至1027-03-20 18:00。",
            f"{name}持有从{end}到{start}的传送通行证，有效期至1027-03-20 18:00，当前有效。",
            f"{name}获准从{start}瞬时前往{PLACES[(i + 1) % 5][1]}，有效期至1027-03-20 18:00。",
            f"{name}曾获从{start}到{end}的跃迁许可，但已于1027-02-20 18:00过期。",
            f"{name}的从{start}到{end}传送权限当前无效。",
        )
        chapter = f"{stamp}，{name}在{start}。\n{stamp}，{name}在{end}。"
        rows.append(case("location_collision", i, "positive", [
            {"name": "permissions.md", "content": positive_permits[i]},
            {"name": "chapter.md", "content": chapter},
        ], issue("location_collision", ("chapter.md", 1), ("chapter.md", 2)), "other actor, reversed/mismatched route, expired or inactive permit"))
        negative_permits = (
            f"管理局向{name}签发瞬时通行许可：允许从{start}抵达{end}，有效期从1027-03-01 00:00到1027-03-20 18:00，当前有效。",
            f"{name}持有有效的传送通行证，可在1027-03-01 00:00至1027-03-20 18:00期间往返{start}与{end}。",
            f"调度记录确认{name}的跃迁权限仍有效，许可路线为{start}至{end}，截至1027-03-20 18:00。",
            f"1027-03-01 00:00起，{name}获准通过折跃门从{start}瞬时前往{end}；许可在1027-03-20 18:00前有效。",
            f"{name}的定向传送许可处于有效状态，覆盖{start}和{end}之间的双向路线，有效期至1027-03-20 18:00。",
        )
        rows.append(case("location_collision", i, "hard-negative", [
            {"name": "permissions.md", "content": negative_permits[i]},
            {"name": "chapter.md", "content": chapter},
        ], [], "applicable actor-route-time mobility permission"))

    sources = (
        "1027-04-01 08:00，{name}阅读来信后获知{fact}。",
        "1027-04-01 08:00，{name}亲眼目击并知道了{fact}。",
        "1027-04-01 08:00，值守官把{fact}告诉{name}。",
        "1027-04-01 08:00，{name}查阅航海档案，得知{fact}。",
        "1027-04-01 08:00，{name}从公告中获知{fact}。",
    )
    for i in range(5):
        name = NAMES[i]
        fact = f"第{i + 1}避风航线位置"
        claim = f"1027-04-01 09:00，{name}准确说出了{fact}。"
        rows.append(case("knowledge_without_acquisition", i, "positive", [
            {"name": "chapter.md", "content": claim + "\n" + sources[i].format(name=name, fact=fact).replace("08:00", "10:00")},
        ], issue("knowledge_without_acquisition", ("chapter.md", 1), ("chapter.md", 2)), "knowledge source occurs after claim"))
        rows.append(case("knowledge_without_acquisition", i, "hard-negative", [
            {"name": "chapter.md", "content": sources[i].format(name=name, fact=fact) + "\n" + claim},
        ], [], "explicit earlier letter, witness, telling, reading or notice source"))

    for i in range(5):
        holder, user = NAMES[i], NAMES[i + 5]
        item = f"潮纹钥匙{i + 1}"
        rows.append(case("item_ownership", i, "positive", [
            {"name": "chapter.md", "content": f"1027-05-01 08:00，{holder}保管{item}。\n1027-05-01 09:00，{user}使用{item}。"},
        ], issue("item_ownership", ("chapter.md", 1), ("chapter.md", 2)), "use by non-holder"))
        rows.append(case("item_ownership", i, "hard-negative", [
            {"name": "chapter.md", "content": f"1027-05-01 08:00，{holder}保管{item}。\n1027-05-01 08:30，{user}获得{item}。\n1027-05-01 09:00，{user}使用{item}。"},
        ], [], "explicit handover before use"))

    for i in range(5):
        actor = NAMES[i]
        other = NAMES[i + 5]
        scope = f"静潮域{i + 1}"
        action = f"回声术{i + 1}式"
        rule = f"在{scope}中，任何{action}都会失效。"
        assertion = f"{actor}却在{scope}中发动{action}。"
        bad_exceptions = (
            f"只有{other}拥有在{scope}使用{action}的例外资格，当前有效。",
            f"{actor}只获准在雾港使用{action}，资格当前有效。",
            f"{actor}拥有在{scope}使用照明术的例外资格，当前有效。",
            f"{actor}在{scope}使用{action}的例外资格已于1027-05-01 00:00过期。",
            f"{actor}在{scope}使用{action}的例外资格当前无效。",
        )
        rows.append(case("world_rule_conflict", i, "positive", [
            {"name": "world.md", "content": rule + "\n" + bad_exceptions[i]},
            {"name": "chapter.md", "content": assertion},
        ], issue("world_rule_conflict", ("world.md", 1), ("chapter.md", 1)), "other actor, wrong scope/action, expired or inactive exception"))
        exception_patterns = (
            f"{actor}拥有在{scope}使用{action}的例外资格，当前有效。",
            f"议会授权{actor}在{scope}发动{action}，豁免状态有效。",
            f"{actor}持有仍生效的规则豁免，可在{scope}施展{action}。",
            f"仅{actor}获准在{scope}使用{action}，该例外当前有效。",
            f"审查记录确认：{actor}在{scope}发动{action}的特别许可处于有效状态。",
        )
        rows.append(case("world_rule_conflict", i, "hard-negative", [
            {"name": "world.md", "content": rule + "\n" + exception_patterns[i]},
            {"name": "chapter.md", "content": assertion},
        ], [], "actor-specific applicable rule exception"))
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "evaluation-challenge-v2"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "test.jsonl"
    rows = build()
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "name": "LoreGuard challenge-v2",
        "dataset_kind": "synthetic natural-language",
        "human_annotated": False,
        "developer_visible": True,
        "blind_test": False,
        "seed": SEED,
        "generator": "scripts/generate_challenge_v2.py",
        "case_count": len(rows),
        "cases_per_category": 10,
        "positive_cases": 25,
        "hard_negative_cases": 25,
        "sha256": digest,
        "limitations": "Fixed-template, developer-visible synthetic challenge. It is not an independent human blind test and cannot estimate production accuracy.",
        "license_note": "Original fictional content; no commercial IP or ConStory-Bench data.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"cases={len(rows)} sha256={digest}")


if __name__ == "__main__":
    main()
