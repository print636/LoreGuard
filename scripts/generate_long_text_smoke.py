from __future__ import annotations

import hashlib
import json
from pathlib import Path


def filler(chapter: int, count: int) -> list[str]:
    return [
        f"第{chapter:02d}卷航海记录第{index:04d}节描述远岸潮声、普通补给清单与不涉及角色状态的背景景物。"
        for index in range(1, count + 1)
    ]


def build() -> dict[str, str]:
    world = [
        "# 静默海域世界档案",
        "林澈又名银羽。",
        "林澈的发色是银色。",
        "1027-06-01 08:00，苏弦保管潮汐钥匙。",
        "进入静默海域后，所有潮汐术都会失效。",
        "1027-06-01 12:00，林澈得知沉星航线入口位置。",
        *filler(0, 140),
    ]
    chapter_one = [
        "# 第一章 远港",
        *filler(1, 145),
        "银羽的发色是黑色。",
    ]
    chapter_two = [
        "# 第二章 双塔记录",
        *filler(2, 145),
        "1027-06-01 10:00，银羽在北港。",
        "1027-06-01 10:00，林澈在南塔。",
    ]
    chapter_three = [
        "# 第三章 门与潮汐",
        *filler(3, 145),
        "1027-06-01 09:00，银羽准确说出了沉星航线入口位置。",
        "1027-06-01 09:30，银羽使用潮汐钥匙。",
        "银羽却在静默海域中发动潮汐术。",
    ]
    return {
        "world.md": "\n".join(world),
        "chapter-01.md": "\n".join(chapter_one),
        "chapter-02.md": "\n".join(chapter_two),
        "chapter-03.md": "\n".join(chapter_three),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "long-text-smoke"
    root.mkdir(parents=True, exist_ok=True)
    documents = build()
    hashes = {}
    for name, content in documents.items():
        path = root / name
        path.write_text(content, encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "dataset_kind": "generated long-text smoke",
        "human_annotated": False,
        "commercial_ip": False,
        "generator": "scripts/generate_long_text_smoke.py",
        "document_count": len(documents),
        "character_count": sum(len(content) for content in documents.values()),
        "expected_categories": [
            "fact_conflict", "location_collision", "knowledge_without_acquisition",
            "item_ownership", "world_rule_conflict",
        ],
        "sha256": hashes,
        "limitations": "Generated smoke fixture for pipeline, evidence-line and regression checks; not an accuracy benchmark.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"documents={len(documents)} chars={manifest['character_count']}")


if __name__ == "__main__":
    main()
