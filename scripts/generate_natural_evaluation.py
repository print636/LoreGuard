from __future__ import annotations

import json
from pathlib import Path


SEED = 20260901
CATEGORIES = (
    "fact_conflict",
    "location_collision",
    "knowledge_without_acquisition",
    "item_ownership",
    "world_rule_conflict",
)
NAMES = ("顾青", "程砚", "黎安", "沈舟", "周岚", "孟川", "季遥", "苏翎", "陆衡", "唐汐")
PLACES = ("东港", "西塔", "雾桥", "星台", "白礁", "石湾", "云庭", "霜堡", "潮阁", "月岛")


def evidence(document: str, line: int = 1) -> dict:
    return {"document": document, "line": line}


def case(
    *,
    category: str,
    index: int,
    positive: bool,
    documents: list[dict],
    expected: list[dict],
    template: str,
) -> dict:
    split = "dev" if index < 4 else "test"
    polarity = "positive" if positive else "hard-negative"
    scenario_id = f"{split}-{category}-{polarity}-{index:02d}"
    return {
        "case_id": scenario_id,
        "scenario_id": scenario_id,
        "split": split,
        "category_focus": category,
        "polarity": polarity,
        "documents": documents,
        "expected_issues": expected,
        "generation": {
            "kind": "synthetic natural-language",
            "template": template,
            "seed": SEED,
        },
    }


def fact_cases(index: int, positive: bool) -> dict:
    name = NAMES[index]
    if positive:
        docs = [
            {"name": "world.md", "content": f"{name}在旧日事故中失去了左臂，之后一直使用机械义肢。"},
            {"name": "chapter.md", "content": f"{name}发现缆绳断裂。他立刻伸出完好的左手抓住缆绳，手臂没有受伤。"},
        ]
        expected = [{"category": "fact_conflict", "evidence": [evidence("world.md"), evidence("chapter.md")]}]
        template = "persistent limb loss versus intact same-side limb"
    elif index % 3 == 0:
        docs = [
            {"name": "world.md", "content": f"{name}曾在事故中失去了右臂，但治疗后右臂已经完全再生。"},
            {"name": "chapter.md", "content": f"{name}进入庭院。她伸出完好的右手接住花瓣。"},
        ]
        expected, template = [], "explicit recovery before intact limb use"
    elif index % 3 == 1:
        docs = [
            {"name": "world.md", "content": f"{name}在战斗中失去了左臂。"},
            {"name": "chapter.md", "content": f"幻象中，{name}伸出看似完好的左手；现实中的身体没有改变。"},
        ]
        expected, template = [], "illusory intact limb is not a physical recovery"
    else:
        docs = [
            {"name": "world.md", "content": f"1024年，{name}的发色是黑色。"},
            {"name": "chapter.md", "content": f"两年后，{name}把头发染成银色，并明确说明这是新的造型。"},
        ]
        expected, template = [], "explicit temporal appearance change"
    return case(category="fact_conflict", index=index, positive=positive, documents=docs, expected=expected, template=template)


def location_cases(index: int, positive: bool) -> dict:
    name, first, second = NAMES[index], PLACES[index], PLACES[(index + 3) % len(PLACES)]
    docs = [{
        "name": "chapter.md",
        "content": (
            f"1026-06-{index + 1:02d} 09:00，航行日志显示{name}仍在{first}仓库盘点物资。\n"
            f"1026-06-{index + 1:02d} 09:00，{name}在{second}顶层与站长会面。"
        ),
    }]
    if positive:
        expected = [{"category": "location_collision", "evidence": [evidence("chapter.md", 1), evidence("chapter.md", 2)]}]
        template = "same character at two ordinary locations at the same timestamp"
    else:
        docs.insert(0, {"name": "world.md", "content": f"持有跃迁许可的人可以在{first}与{second}之间瞬时移动；{name}持有有效许可。"})
        expected, template = [], "explicit licensed teleport exception"
    return case(category="location_collision", index=index, positive=positive, documents=docs, expected=expected, template=template)


def knowledge_cases(index: int, positive: bool) -> dict:
    name = NAMES[index]
    secret = f"第{index + 1}号航道入口位置"
    if positive:
        docs = [
            {"name": "world.md", "content": f"1026-07-{index + 1:02d} 18:00，档案官才第一次把{secret}告诉{name}，此前{name}不知道该位置。"},
            {"name": "chapter.md", "content": f"1026-07-{index + 1:02d} 09:00，{name}对同行者准确说出了{secret}，并要求立即转向。"},
        ]
        expected = [{"category": "knowledge_without_acquisition", "evidence": [evidence("world.md"), evidence("chapter.md")]}]
        template = "claim before first explicit acquisition"
    else:
        docs = [
            {"name": "letter.md", "content": f"1026-07-{index + 1:02d} 08:00，{name}通过加密信件得知{secret}，并确认了寄件人身份。"},
            {"name": "chapter.md", "content": f"1026-07-{index + 1:02d} 09:00，{name}对同行者准确说出了{secret}。"},
        ]
        expected, template = [], "knowledge acquired earlier through an authenticated letter"
    return case(category="knowledge_without_acquisition", index=index, positive=positive, documents=docs, expected=expected, template=template)


def item_cases(index: int, positive: bool) -> dict:
    user, owner, item = NAMES[index], NAMES[(index + 4) % len(NAMES)], f"赤曜章{index + 1}号"
    docs = [{"name": "world.md", "content": f"城议会保存着唯一一枚“{item}”。徽章一直由{owner}保管。"}]
    if positive:
        docs.append({"name": "chapter.md", "content": f"{user}进入档案室。\n他从匣中取出{item}，并在没有交接文书的情况下盖下通行印记。"})
        expected = [{"category": "item_ownership", "evidence": [evidence("world.md"), evidence("chapter.md", 2)]}]
        template = "use of a uniquely held item without transfer"
    else:
        docs.extend([
            {"name": "transfer.md", "content": f"1026-08-{index + 1:02d} 08:00，{user}获得{item}，双方完成公开交接。"},
            {"name": "chapter.md", "content": f"1026-08-{index + 1:02d} 09:00，{user}使用{item}打开档案柜。"},
        ])
        expected, template = [], "valid transfer before item use"
    return case(category="item_ownership", index=index, positive=positive, documents=docs, expected=expected, template=template)


def rule_cases(index: int, positive: bool) -> dict:
    name, scope, ability = NAMES[index], f"静音区{index + 1}号", f"声波术{index + 1}式"
    exception = f"持有豁免徽记的术士除外；{name}持有有效徽记。" if not positive else ""
    docs = [
        {"name": "world.md", "content": f"进入{scope}后，所有{ability}都会失效。{exception}"},
        {"name": "chapter.md", "content": f"{name}在{scope}中央仍发动{ability}，震开了石门。"},
    ]
    if positive:
        expected = [{"category": "world_rule_conflict", "evidence": [evidence("world.md"), evidence("chapter.md")]}]
        template = "ability performed inside a disabling scope"
    else:
        expected, template = [], "explicit holder-specific exception to disabling rule"
    return case(category="world_rule_conflict", index=index, positive=positive, documents=docs, expected=expected, template=template)


def generate() -> tuple[list[dict], list[dict]]:
    builders = (fact_cases, location_cases, knowledge_cases, item_cases, rule_cases)
    cases = [builder(index, positive) for builder in builders for positive in (True, False) for index in range(10)]
    dev = [row for row in cases if row["split"] == "dev"]
    test = [row for row in cases if row["split"] == "test"]
    return dev, test


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "evaluation-natural"
    root.mkdir(parents=True, exist_ok=True)
    dev, test = generate()
    write_jsonl(root / "dev.jsonl", dev)
    write_jsonl(root / "test.jsonl", test)
    manifest = {
        "dataset_kind": "synthetic natural-language",
        "seed": SEED,
        "generator": "scripts/generate_natural_evaluation.py",
        "license_note": "Original fictional material generated for LoreGuard; no commercial IP or ConStory-Bench data.",
        "dev_cases": len(dev),
        "test_cases": len(test),
        "scenario_overlap": False,
        "categories": list(CATEGORIES),
        "limitations": "Template-generated and not human-annotated; metrics do not estimate production accuracy.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"dev={len(dev)} test={len(test)} total={len(dev) + len(test)}")


if __name__ == "__main__":
    main()
