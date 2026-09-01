from __future__ import annotations

import hashlib
import json
from pathlib import Path


DATASET_VERSION = "3.0.0"
WORLD = "潮痕群岛"


def document(name: str, *lines: str) -> dict:
    return {"name": name, "content": "\n".join(lines)}


def expected(category: str, *evidence: tuple[str, int]) -> dict:
    return {
        "category": category,
        "evidence": [
            {"document": document_name, "line": line}
            for document_name, line in evidence
        ],
    }


def case(
    case_id: str,
    category_focus: str,
    polarity: str,
    documents: list[dict],
    expected_issues: list[dict],
    capability: str,
    difficulty: str,
) -> dict:
    return {
        "case_id": case_id,
        "scenario_id": case_id,
        "split": "test",
        "category_focus": category_focus,
        "polarity": polarity,
        "documents": documents,
        "expected_issues": expected_issues,
        "generation": {
            "kind": "developer-authored fixed natural-language scenario",
            "dataset_version": DATASET_VERSION,
            "developer_visible": True,
            "human_annotated": False,
            "blind_test": False,
            "world": WORLD,
            "capability": capability,
            "difficulty": difficulty,
        },
    }


def build() -> list[dict]:
    shared_context = document(
        "setting.md",
        "潮痕群岛依靠潮核维持航路，岛间公文使用群岛历纪年。",
        "雾港是月沫城的外港，中央档案厅位于月沫城内城区。",
    )

    rows = [
        case(
            "complex-v3-integrated-conflicts",
            "fact_conflict",
            "positive",
            [
                shared_context,
                document(
                    "canon.md",
                    "在黑潮静默区中，任何回声测绘术都会失效。",
                    "“潮汐钥”一直由洛岑保管。",
                    "云栖的航籍状态是内港注册。",
                ),
                document(
                    "knowledge-ledger.md",
                    "1044-06-18 10:20，纪员第一次把沉钟库第三入口位置告诉桑砚。",
                ),
                document(
                    "chapter-01.md",
                    "1044-06-18 09:10，桑砚准确说出了沉钟库第三入口位置。",
                    "桑砚没有解释坐标从何而来，便带队驶向旧航道。",
                    "1044-06-18 09:30，莫行取出潮汐钥，并按下启动机关。",
                ),
                document(
                    "chapter-02.md",
                    "1044-06-18 11:00，云栖正在沉钟港北防波堤检查封印。",
                    "1044-06-18 11:00，巡逻记录显示云栖仍在月沫城中央档案厅签署封港令。",
                    "云栖的航籍状态是外海通缉。",
                    "云栖却在黑潮静默区中发动回声测绘术。",
                ),
            ],
            [
                expected("world_rule_conflict", ("canon.md", 1), ("chapter-02.md", 4)),
                expected("item_ownership", ("canon.md", 2), ("chapter-01.md", 3)),
                expected("fact_conflict", ("canon.md", 3), ("chapter-02.md", 3)),
                expected("knowledge_without_acquisition", ("chapter-01.md", 1), ("knowledge-ledger.md", 1)),
                expected("location_collision", ("chapter-02.md", 1), ("chapter-02.md", 2)),
            ],
            "five-category cross-document integrated failure",
            "Five interacting issue families are mixed with narrative filler across five files.",
        ),
        case(
            "complex-v3-fact-canonical-conflict",
            "fact_conflict",
            "positive",
            [
                shared_context,
                document("canon.md", "岑烁的离港权限是冻结状态。"),
                document("chapter-01.md", "岑烁仍留在审查室等待复核。"),
                document("chapter-02.md", "岑烁的离港权限是生效状态。"),
            ],
            [expected("fact_conflict", ("canon.md", 1), ("chapter-02.md", 1))],
            "canonical administrative state conflict",
            "The contradiction affects plot access rather than a cosmetic attribute.",
        ),
        case(
            "complex-v3-fact-state-transition",
            "fact_conflict",
            "hard-negative",
            [
                shared_context,
                document("chapter-01.md", "1044-06-18 08:00，岑烁的离港权限是冻结状态。"),
                document("orders.md", "1044-06-18 12:00，议会通过复核并解除岑烁的限制。"),
                document("chapter-02.md", "1044-06-18 12:30，岑烁的离港权限是生效状态。"),
            ],
            [],
            "fact state transition",
            "Same subject and predicate change after an explicit intervening order; this is not a contradiction.",
        ),
        case(
            "complex-v3-location-collision",
            "location_collision",
            "positive",
            [
                shared_context,
                document("travel-ledger.md", "本日未向祁霁签发折跃或传送许可。"),
                document("chapter-01.md", "1044-06-18 14:00，祁霁正在沉钟港北防波堤检查潮压。"),
                document("chapter-02.md", "1044-06-18 14:00，记录显示祁霁仍在月沫城中央档案厅签署航令。"),
            ],
            [expected("location_collision", ("chapter-01.md", 1), ("chapter-02.md", 1))],
            "precise timeline collision",
            "Two distant places are stated in separate chapters at the same minute.",
        ),
        case(
            "complex-v3-location-nested-place",
            "location_collision",
            "hard-negative",
            [
                shared_context,
                document("map.md", "中央档案厅位于月沫城内城区，属于同一连续地点。"),
                document("chapter-01.md", "1044-06-18 14:00，祁霁正在月沫城调查失窃案。"),
                document("chapter-02.md", "1044-06-18 14:00，记录显示祁霁仍在月沫城中央档案厅查阅卷宗。"),
            ],
            [],
            "nested location hierarchy",
            "A city and a room inside that city are compatible descriptions, not simultaneous distant presence.",
        ),
        case(
            "complex-v3-location-valid-transport",
            "location_collision",
            "hard-negative",
            [
                shared_context,
                document("permits.md", "议会向祁霁签发瞬时通行许可，路线为沉钟港至月沫城，当前有效。"),
                document("chapter-01.md", "1044-06-18 14:00，祁霁正在沉钟港北防波堤检查潮压。"),
                document("chapter-02.md", "1044-06-18 14:00，记录显示祁霁仍在月沫城中央档案厅签署航令。"),
            ],
            [],
            "actor-bound transport permission with nested endpoints",
            "An active permission covers the actor and both parent locations.",
        ),
        case(
            "complex-v3-knowledge-premature",
            "knowledge_without_acquisition",
            "positive",
            [
                shared_context,
                document("access-policy.md", "沉钟库第三入口位置只记录在未公开的密卷中。"),
                document("chapter-01.md", "1044-06-18 09:10，桑砚准确说出了沉钟库第三入口位置。"),
                document("chapter-02.md", "1044-06-18 10:20，纪员第一次把沉钟库第三入口位置告诉桑砚。"),
            ],
            [expected("knowledge_without_acquisition", ("chapter-01.md", 1), ("chapter-02.md", 1))],
            "knowledge provenance ordering",
            "The only explicit acquisition occurs after the precise claim.",
        ),
        case(
            "complex-v3-knowledge-source-chain",
            "knowledge_without_acquisition",
            "hard-negative",
            [
                shared_context,
                document("chapter-01.md", "1044-06-18 08:20，桑砚阅读了密函后得知沉钟库第三入口位置。"),
                document("chapter-02.md", "1044-06-18 09:10，桑砚准确说出了沉钟库第三入口位置。"),
                document("audit.md", "密函在八时前由档案官签发，来源链完整。"),
            ],
            [],
            "knowledge source chain",
            "An explicit earlier letter source justifies the later claim.",
        ),
        case(
            "complex-v3-knowledge-incomplete-ledger",
            "knowledge_without_acquisition",
            "hard-negative",
            [
                shared_context,
                document("chapter-01.md", "1044-06-18 09:10，桑砚准确说出了沉钟库第三入口位置。"),
                document("chapter-02.md", "队伍按坐标抵达封闭航道，没有交代消息来源。"),
                document("audit.md", "现存档案无法证明桑砚何时第一次接触该位置。"),
            ],
            [],
            "absence of acquisition evidence",
            "Missing provenance is incomplete information, not proof that the character learned it later.",
        ),
        case(
            "complex-v3-item-unauthorized-use",
            "item_ownership",
            "positive",
            [
                shared_context,
                document("inventory.md", "“潮汐钥”一直由洛岑保管。"),
                document("chapter-01.md", "洛岑把钥匙锁进独立保管柜，没有办理交接。"),
                document("chapter-02.md", "1044-06-18 09:30，莫行取出潮汐钥，并按下启动机关。"),
            ],
            [expected("item_ownership", ("inventory.md", 1), ("chapter-02.md", 1))],
            "item custody violation",
            "A plot-critical key is used by another actor without any transfer record.",
        ),
        case(
            "complex-v3-item-authorized-transfer",
            "item_ownership",
            "hard-negative",
            [
                shared_context,
                document("inventory.md", "1044-06-18 08:00，洛岑保管潮汐钥。"),
                document("handover.md", "1044-06-18 09:00，莫行获得潮汐钥。"),
                document("chapter-02.md", "1044-06-18 09:30，莫行使用潮汐钥。"),
            ],
            [],
            "item transfer and authorization",
            "The latest custody record authorizes the later user.",
        ),
        case(
            "complex-v3-rule-scope-violation",
            "world_rule_conflict",
            "positive",
            [
                shared_context,
                document("physics.md", "在黑潮静默区中，任何回声测绘术都会失效。"),
                document("exceptions.md", "当前没有向闻序签发黑潮静默区规则豁免。"),
                document("chapter-02.md", "闻序却在黑潮静默区中发动回声测绘术。"),
            ],
            [expected("world_rule_conflict", ("physics.md", 1), ("chapter-02.md", 1))],
            "world-rule scoped action violation",
            "The action, scope, and actor are explicit; no applicable exception exists.",
        ),
        case(
            "complex-v3-rule-actor-exception",
            "world_rule_conflict",
            "hard-negative",
            [
                shared_context,
                document("physics.md", "在黑潮静默区中，任何回声测绘术都会失效。"),
                document("exceptions.md", "议会授权闻序在黑潮静默区发动回声测绘术，豁免状态有效。"),
                document("chapter-02.md", "闻序却在黑潮静默区中发动回声测绘术。"),
            ],
            [],
            "actor-specific world-rule exception",
            "The exception matches actor, scope, action, and active status.",
        ),
        case(
            "complex-v3-rule-different-scope",
            "world_rule_conflict",
            "hard-negative",
            [
                shared_context,
                document("physics.md", "在黑潮静默区中，任何回声测绘术都会失效。"),
                document("map.md", "晴潮观测站位于黑潮静默区边界之外。"),
                document("chapter-02.md", "闻序在晴潮观测站中发动回声测绘术。"),
            ],
            [],
            "world-rule scope boundary",
            "The same action outside the restricted scope is legal.",
        ),
    ]
    return rows


def validate(rows: list[dict]) -> None:
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case ids must be unique")
    required_categories = {
        "fact_conflict",
        "location_collision",
        "knowledge_without_acquisition",
        "item_ownership",
        "world_rule_conflict",
    }
    if {row["category_focus"] for row in rows} != required_categories:
        raise ValueError("all five issue categories must be covered")
    for row in rows:
        if len(row["documents"]) < 4:
            raise ValueError(f"{row['case_id']} must span at least four documents")


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "evaluation-complex-v3"
    root.mkdir(parents=True, exist_ok=True)
    rows = build()
    validate(rows)
    target = root / "test.jsonl"
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    target.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    expected_count = sum(len(row["expected_issues"]) for row in rows)
    categories = sorted({row["category_focus"] for row in rows})
    manifest = {
        "name": "LoreGuard complex original acceptance set v3",
        "version": DATASET_VERSION,
        "dataset_kind": "developer-authored fixed natural-language scenarios",
        "world": WORLD,
        "human_annotated": False,
        "developer_visible": True,
        "blind_test": False,
        "model_required": False,
        "generator": "scripts/generate_complex_v3.py",
        "case_count": len(rows),
        "positive_case_count": sum(bool(row["expected_issues"]) for row in rows),
        "hard_negative_case_count": sum(not row["expected_issues"] for row in rows),
        "expected_issue_count": expected_count,
        "categories": categories,
        "minimum_documents_per_case": min(len(row["documents"]) for row in rows),
        "evidence_contract": "Every expected issue fixes the exact source document and 1-based line number pair.",
        "sha256": digest,
        "limitations": (
            "Developer-authored and developer-visible synthetic acceptance data. "
            "It is not independently annotated, not blind, and cannot estimate production accuracy."
        ),
        "license_note": "Original fictional content; no commercial IP or ConStory-Bench content.",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"cases={len(rows)} expected_issues={expected_count} sha256={digest}")


if __name__ == "__main__":
    main()
