from types import SimpleNamespace
import unittest

from app.projections import project_graph, project_timeline


def evidence(line: int, *, document_id: str = "doc", name: str = "chapter.md") -> dict:
    return {
        "document_id": document_id,
        "document_name": name,
        "line_start": line,
        "line_end": line,
        "text": f"第 {line} 行证据",
    }


def record(identifier: str, kind: str, attrs: dict[str, str], line: int):
    return SimpleNamespace(id=identifier, kind=kind, attrs=attrs, evidence=evidence(line))


class VisualizationProjectionTests(unittest.TestCase):
    def test_all_record_families_project_to_typed_graph_with_evidence_links(self):
        records = [
            record("r0", "entity", {"name": "林澈", "kind": "character"}, 1),
            record("r1", "fact", {"subject": "林澈", "predicate": "发色", "value": "银色"}, 2),
            record("r2", "event", {"participants": "林澈,苏弦", "location": "北港", "time": "1026-04-03 10:00"}, 3),
            record("r3", "knows", {"character": "林澈", "fact": "星门口令", "time": "1026-04-03 12:00"}, 4),
            record("r4", "claims_knows", {"character": "苏弦", "fact": "星门口令", "time": "此前"}, 5),
            record("r5", "item", {"owner": "苏弦", "item": "星门钥匙", "time": "1026-04-03 08:00"}, 6),
            record("r6", "uses", {"user": "林澈", "item": "星门钥匙", "time": "1026-04-03 09:30"}, 7),
            record("r7", "world_rule", {"key": "星门动力", "value": "潮汐晶核"}, 8),
            record("r8", "world_assert", {"key": "星门动力", "value": "普通火焰", "actor": "林澈"}, 9),
        ]
        issue = SimpleNamespace(id="issue-1", evidence=[evidence(3)])

        graph = project_graph("run", records, [issue])

        self.assertFalse(graph.truncated)
        self.assertEqual(
            {"fact", "event", "knows", "claims_knows", "item", "uses", "world_rule", "world_assert"},
            {edge.type for edge in graph.edges},
        )
        event_edges = [edge for edge in graph.edges if edge.type == "event"]
        self.assertEqual(2, len(event_edges))
        self.assertTrue(all(edge.issue_ids == ["issue-1"] for edge in event_edges))
        self.assertTrue(all(edge.evidence.text for edge in graph.edges))
        assertion = next(edge for edge in graph.edges if edge.type == "world_assert")
        source = next(node for node in graph.nodes if node.id == assertion.source)
        self.assertEqual(("entity", "林澈"), (source.type, source.label))

    def test_typed_namespaces_do_not_merge_equal_labels(self):
        records = [
            record("r1", "fact", {"subject": "北港", "predicate": "状态", "value": "封锁"}, 1),
            record("r2", "event", {"participants": "林澈", "location": "北港", "time": "1026-04-03 10:00"}, 2),
        ]
        graph = project_graph("run", records, [])
        matching = [node for node in graph.nodes if node.label == "北港"]
        self.assertEqual({"entity", "location"}, {node.type for node in matching})
        self.assertEqual(2, len({node.id for node in matching}))

    def test_timeline_sorts_only_explicit_calendar_times(self):
        records = [
            record("relative", "claims_knows", {"character": "林澈", "fact": "入口", "time": "此前"}, 1),
            record("late", "event", {"participants": "林澈", "location": "南塔", "time": "1026-04-03T10:00"}, 2),
            record("early", "item", {"owner": "苏弦", "item": "钥匙", "time": "1026-04-03 08:00"}, 3),
            record("untimed", "world_rule", {"key": "星门动力", "value": "晶核"}, 4),
            record("same", "uses", {"user": "苏弦", "item": "钥匙", "time": "1026-04-03 10:00"}, 5),
        ]

        timeline = project_timeline("run", records, [])

        self.assertEqual(["1026-04-03 08:00", "1026-04-03 10:00"], [group.timestamp for group in timeline.groups])
        self.assertEqual(2, len(timeline.groups[1].entries))
        self.assertEqual(["relative", "unknown"], [entry.precision for entry in timeline.unscheduled])
        self.assertTrue(timeline.warnings)

    def test_graph_limits_are_enforced_with_warning(self):
        records = [
            record(f"r{index}", "event", {"participants": f"角色{index}", "location": f"地点{index}", "time": "1026-04-03 10:00"}, index)
            for index in range(1, 8)
        ]
        graph = project_graph("run", records, [], node_limit=4, edge_limit=2)
        self.assertTrue(graph.truncated)
        self.assertLessEqual(len(graph.nodes), 4)
        self.assertLessEqual(len(graph.edges), 2)
        self.assertIn("上限截断", graph.warnings[0])


if __name__ == "__main__":
    unittest.main()
