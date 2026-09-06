from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path


SUITE_ROOT = Path(__file__).resolve().parents[1] / "data" / "evidence-retrieval-v1"
MANIFEST_PATH = SUITE_ROOT / "manifest.json"
FREEZE_PATH = SUITE_ROOT / "freeze.json"


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"expected an object in {path.name}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cjk_bigrams(text: str) -> set[str]:
    compact = "".join(character for character in text if "\u4e00" <= character <= "\u9fff")
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def _normalized_cjk(text: str) -> str:
    return "".join(character for character in text if "\u4e00" <= character <= "\u9fff")


def _longest_common_substring(left: str, right: str) -> int:
    left = _normalized_cjk(left)
    right = _normalized_cjk(right)
    previous = [0] * (len(right) + 1)
    best = 0
    for left_character in left:
        current = [0]
        for index, right_character in enumerate(right, start=1):
            length = previous[index - 1] + 1 if left_character == right_character else 0
            current.append(length)
            best = max(best, length)
        previous = current
    return best


class EvidenceRetrievalV1ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = _load_json(MANIFEST_PATH)
        cls.freeze = _load_json(FREEZE_PATH)
        cls.worlds = {
            world["world_id"]: world for world in cls.manifest.get("worlds", [])
        }
        cls.projects = {
            world["project_id"]: world for world in cls.manifest.get("worlds", [])
        }
        cls.documents = {
            (world["world_id"], document["document_id"], document["version"]): document
            for world in cls.manifest.get("worlds", [])
            for document in world.get("documents", [])
        }

    def _safe_path(self, relative_path: str) -> Path:
        self.assertIsInstance(relative_path, str)
        self.assertNotEqual(relative_path.strip(), "")
        candidate = (SUITE_ROOT / relative_path).resolve()
        candidate.relative_to(SUITE_ROOT.resolve())
        self.assertTrue(candidate.is_file(), relative_path)
        return candidate

    def _evidence_text(self, task: dict) -> str:
        sections: list[str] = []
        for evidence in task["expected_evidence"]:
            document = self.documents[
                (task["world_id"], evidence["document_id"], evidence["version"])
            ]
            lines = self._safe_path(document["path"]).read_text(
                encoding="utf-8"
            ).splitlines()
            sections.extend(lines[evidence["start_line"] - 1 : evidence["end_line"]])
        return "\n".join(sections)

    def test_frozen_hashes_cover_manifest_and_every_document(self) -> None:
        self.assertEqual("1.0", self.freeze.get("schema_version"))
        self.assertEqual(self.manifest.get("dataset_id"), self.freeze.get("dataset_id"))

        frozen_manifest = self.freeze.get("manifest")
        self.assertIsInstance(frozen_manifest, dict)
        self.assertEqual("manifest.json", frozen_manifest.get("path"))
        self.assertEqual(_sha256(MANIFEST_PATH), frozen_manifest.get("sha256"))
        self.assertEqual(MANIFEST_PATH.stat().st_size, frozen_manifest.get("bytes"))
        self.assertEqual(
            len(MANIFEST_PATH.read_text(encoding="utf-8").splitlines()),
            frozen_manifest.get("line_count"),
        )

        frozen_documents = self.freeze.get("frozen_documents")
        self.assertIsInstance(frozen_documents, list)
        frozen_by_path = {entry["path"]: entry for entry in frozen_documents}
        manifest_paths = {
            document["path"]
            for world in self.manifest["worlds"]
            for document in world["documents"]
        }
        disk_paths = {
            path.relative_to(SUITE_ROOT).as_posix()
            for path in (SUITE_ROOT / "worlds").rglob("*.md")
        }
        self.assertEqual(manifest_paths, disk_paths)
        self.assertEqual(manifest_paths, set(frozen_by_path))

        for relative_path, frozen in frozen_by_path.items():
            with self.subTest(path=relative_path):
                path = self._safe_path(relative_path)
                lines = path.read_text(encoding="utf-8").splitlines()
                self.assertTrue(lines)
                self.assertTrue(all(line.strip() for line in lines))
                self.assertEqual(_sha256(path), frozen.get("sha256"))
                self.assertEqual(path.stat().st_size, frozen.get("bytes"))
                self.assertEqual(len(lines), frozen.get("line_count"))

    def test_worlds_are_strictly_isolated_between_dev_and_holdout(self) -> None:
        self.assertEqual(
            {
                "schema_version",
                "dataset_id",
                "language",
                "status",
                "license",
                "provenance",
                "evaluation_boundary",
                "target_profile_id",
                "profiles",
                "worlds",
                "tasks",
            },
            set(self.manifest),
        )
        self.assertEqual("1.0", self.manifest.get("schema_version"))
        self.assertEqual("evidence-retrieval-v1", self.manifest.get("dataset_id"))
        self.assertEqual("zh-CN", self.manifest.get("language"))
        self.assertEqual("frozen", self.manifest.get("status"))
        self.assertEqual(4, len(self.worlds))
        self.assertEqual(4, len(self.projects))

        profiles = self.manifest["profiles"]
        self.assertEqual(
            len(profiles), len({profile["profile_id"] for profile in profiles})
        )
        for profile in profiles:
            self.assertEqual(
                {
                    "profile_id",
                    "model_identifier",
                    "model_revision",
                    "dimensions",
                    "normalized",
                },
                set(profile),
            )
            self.assertIs(type(profile["dimensions"]), int)
            self.assertGreater(profile["dimensions"], 0)
            self.assertIs(type(profile["normalized"]), bool)

        worlds_by_split: dict[str, set[str]] = {"dev": set(), "holdout": set()}
        for world_id, world in self.worlds.items():
            self.assertEqual(
                {"world_id", "project_id", "split", "documents"}, set(world)
            )
            self.assertIn(world["split"], worlds_by_split)
            worlds_by_split[world["split"]].add(world_id)
            documents_by_id: dict[str, list[dict]] = {}
            for document in world["documents"]:
                self.assertEqual(
                    {"document_id", "version", "role", "active", "path"},
                    set(document),
                )
                self.assertIs(type(document["version"]), int)
                self.assertGreater(document["version"], 0)
                self.assertIn(document["role"], {"canon", "chapter", "isolation_decoy"})
                self.assertIs(type(document["active"]), bool)
                self._safe_path(document["path"])
                documents_by_id.setdefault(document["document_id"], []).append(document)
            self.assertTrue(documents_by_id)
            for versions in documents_by_id.values():
                self.assertEqual(len(versions), len({item["version"] for item in versions}))
                self.assertEqual(1, sum(item["active"] for item in versions))

        self.assertTrue(worlds_by_split["dev"])
        self.assertGreaterEqual(len(worlds_by_split["holdout"]), 2)
        self.assertTrue(worlds_by_split["dev"].isdisjoint(worlds_by_split["holdout"]))

    def test_task_schema_evidence_ranges_and_required_coverage(self) -> None:
        tasks = self.manifest.get("tasks")
        self.assertIsInstance(tasks, list)
        self.assertGreaterEqual(len(tasks), 40)
        self.assertGreaterEqual(sum(task["split"] == "holdout" for task in tasks), 24)

        task_ids = [task["task_id"] for task in tasks]
        queries = [task["query"] for task in tasks]
        self.assertEqual(len(task_ids), len(set(task_ids)))
        self.assertEqual(len(queries), len(set(queries)))

        label_counts: Counter[str] = Counter()
        for task in tasks:
            with self.subTest(task_id=task.get("task_id")):
                self.assertIn(
                    set(task),
                    (
                        {
                            "task_id",
                            "split",
                            "world_id",
                            "query",
                            "difficulty",
                            "labels",
                            "expected_evidence",
                        },
                        {
                            "task_id",
                            "split",
                            "world_id",
                            "query",
                            "difficulty",
                            "labels",
                            "expected_evidence",
                            "negative_candidates",
                        },
                    ),
                )
                self.assertRegex(task["task_id"], r"^[A-Z]{2}-[DH]-\d{2}$")
                self.assertIn(task["difficulty"], {"easy", "medium", "hard"})
                self.assertEqual(task["split"], self.worlds[task["world_id"]]["split"])
                expected_split_marker = "-D-" if task["split"] == "dev" else "-H-"
                self.assertIn(expected_split_marker, task["task_id"])
                self.assertIsInstance(task["query"], str)
                self.assertGreaterEqual(len(task["query"].strip()), 8)
                self.assertGreaterEqual(len(_normalized_cjk(task["query"])), 8)
                self.assertIsInstance(task["labels"], list)
                self.assertEqual(len(task["labels"]), len(set(task["labels"])))
                self.assertTrue(task["labels"])
                label_counts.update(task["labels"])

                expected = task["expected_evidence"]
                self.assertIsInstance(expected, list)
                self.assertTrue(expected)
                expected_document_ids = {
                    evidence["document_id"] for evidence in expected
                }
                if "cross_chapter_causality" in task["labels"]:
                    self.assertGreaterEqual(len(expected_document_ids), 2)
                if "knowledge_acquisition" in task["labels"]:
                    self.assertGreaterEqual(len(expected_document_ids), 2)
                for evidence in expected:
                    self.assertEqual(
                        {"document_id", "version", "start_line", "end_line"},
                        set(evidence),
                    )
                    key = (
                        task["world_id"],
                        evidence["document_id"],
                        evidence["version"],
                    )
                    self.assertIn(key, self.documents)
                    document = self.documents[key]
                    self.assertTrue(document["active"])
                    lines = self._safe_path(document["path"]).read_text(
                        encoding="utf-8"
                    ).splitlines()
                    self.assertIs(type(evidence["start_line"]), int)
                    self.assertIs(type(evidence["end_line"]), int)
                    self.assertLessEqual(1, evidence["start_line"])
                    self.assertLessEqual(evidence["start_line"], evidence["end_line"])
                    self.assertLessEqual(evidence["end_line"], len(lines))

                evidence_text = self._evidence_text(task)
                self.assertLessEqual(
                    _longest_common_substring(task["query"], evidence_text),
                    7,
                    "query mechanically copies a long answer phrase",
                )
                if "low_lexical_overlap" in task["labels"]:
                    query_bigrams = _cjk_bigrams(task["query"])
                    overlap = len(query_bigrams & _cjk_bigrams(evidence_text)) / max(
                        1, len(query_bigrams)
                    )
                    self.assertLessEqual(overlap, 0.25)

        required_minimums = {
            "low_lexical_overlap": 12,
            "alias": 4,
            "exception_rule": 6,
            "cross_chapter_causality": 6,
            "knowledge_acquisition": 4,
            "isolation_negative": 3,
        }
        for label, minimum in required_minimums.items():
            with self.subTest(label=label):
                self.assertGreaterEqual(label_counts[label], minimum)

        expected_counts = self.freeze["counts"]
        self.assertEqual(len(tasks), expected_counts["tasks"])
        self.assertEqual(
            sum(task["split"] == "dev" for task in tasks),
            expected_counts["dev_tasks"],
        )
        self.assertEqual(
            sum(task["split"] == "holdout" for task in tasks),
            expected_counts["holdout_tasks"],
        )
        self.assertEqual(len(self.worlds), expected_counts["worlds"])
        self.assertEqual(
            label_counts["low_lexical_overlap"],
            expected_counts["low_lexical_overlap_tasks"],
        )

    def test_isolation_negatives_are_real_and_outside_the_allowed_snapshot(self) -> None:
        target_profile = self.manifest["target_profile_id"]
        profile_ids = {profile["profile_id"] for profile in self.manifest["profiles"]}
        self.assertIn(target_profile, profile_ids)
        negative_kind_counts: Counter[str] = Counter()

        for task in self.manifest["tasks"]:
            negatives = task.get("negative_candidates", [])
            if negatives:
                self.assertIn("isolation_negative", task["labels"])
            expected_refs = {
                (evidence["document_id"], evidence["version"])
                for evidence in task["expected_evidence"]
            }
            task_world = self.worlds[task["world_id"]]
            for negative in negatives:
                self.assertEqual(
                    {
                        "kind",
                        "project_id",
                        "document_id",
                        "version",
                        "start_line",
                        "end_line",
                        "profile_id",
                    },
                    set(negative),
                )
                kind = negative["kind"]
                negative_kind_counts[kind] += 1
                self.assertIn(negative["profile_id"], profile_ids)
                self.assertIn(negative["project_id"], self.projects)
                negative_world = self.projects[negative["project_id"]]
                self.assertEqual(task["split"], negative_world["split"])
                key = (
                    negative_world["world_id"],
                    negative["document_id"],
                    negative["version"],
                )
                self.assertIn(key, self.documents)
                document = self.documents[key]
                negative_lines = self._safe_path(document["path"]).read_text(
                    encoding="utf-8"
                ).splitlines()
                self.assertIs(type(negative["start_line"]), int)
                self.assertIs(type(negative["end_line"]), int)
                self.assertLessEqual(1, negative["start_line"])
                self.assertLessEqual(negative["start_line"], negative["end_line"])
                self.assertLessEqual(negative["end_line"], len(negative_lines))
                negative_text = "\n".join(
                    negative_lines[
                        negative["start_line"] - 1 : negative["end_line"]
                    ]
                )

                if kind == "stale_version":
                    self.assertEqual(task_world["project_id"], negative["project_id"])
                    self.assertFalse(document["active"])
                    self.assertEqual(target_profile, negative["profile_id"])
                elif kind == "wrong_profile":
                    self.assertEqual(task_world["project_id"], negative["project_id"])
                    self.assertTrue(document["active"])
                    self.assertNotEqual(target_profile, negative["profile_id"])
                elif kind == "wrong_project":
                    self.assertNotEqual(task_world["project_id"], negative["project_id"])
                    self.assertEqual(target_profile, negative["profile_id"])
                    self.assertTrue(_cjk_bigrams(task["query"]) & _cjk_bigrams(negative_text))
                elif kind == "wrong_document":
                    self.assertEqual(task_world["project_id"], negative["project_id"])
                    self.assertTrue(document["active"])
                    self.assertNotIn(
                        (negative["document_id"], negative["version"]), expected_refs
                    )
                    self.assertEqual(target_profile, negative["profile_id"])
                    self.assertTrue(_cjk_bigrams(task["query"]) & _cjk_bigrams(negative_text))
                else:
                    self.fail(f"unknown negative kind: {kind}")

        for required_kind in (
            "stale_version",
            "wrong_profile",
            "wrong_document",
            "wrong_project",
        ):
            with self.subTest(kind=required_kind):
                self.assertGreaterEqual(negative_kind_counts[required_kind], 1)


if __name__ == "__main__":
    unittest.main()
