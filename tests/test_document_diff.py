import unittest

from app.document_diff import build_document_diff


class DocumentDiffTests(unittest.TestCase):
    def test_line_diff_summary_and_line_numbers(self):
        result = build_document_diff(
            "第一行\n旧内容\n保留行\n",
            "第一行\n新内容\n保留行\n新增行\n",
        )
        self.assertEqual(2, result["summary"]["added_lines"])
        self.assertEqual(1, result["summary"]["removed_lines"])
        self.assertEqual(2, result["summary"]["unchanged_lines"])
        lines = [line for hunk in result["hunks"] for line in hunk["lines"]]
        removed = next(line for line in lines if line["type"] == "removed")
        added = [line for line in lines if line["type"] == "added"]
        self.assertEqual((2, None, "旧内容"), (removed["old_line"], removed["new_line"], removed["content"]))
        self.assertEqual([2, 4], [line["new_line"] for line in added])
        self.assertEqual([], result["warnings"])

    def test_large_input_and_output_are_explicitly_truncated(self):
        result = build_document_diff(
            "\n".join(f"旧-{index}" for index in range(20)),
            "\n".join(f"新-{index}" for index in range(20)),
            max_lines=8,
            max_chars=10_000,
            max_output_lines=5,
        )
        self.assertTrue(result["summary"]["input_truncated"])
        self.assertTrue(result["summary"]["output_truncated"])
        self.assertEqual(8, result["summary"]["compared_old_lines"])
        self.assertEqual(20, result["summary"]["old_total_lines"])
        self.assertEqual(2, len(result["warnings"]))
        self.assertEqual(5, sum(len(hunk["lines"]) for hunk in result["hunks"]))

    def test_identical_content_has_no_hunks(self):
        result = build_document_diff("相同\n文本", "相同\n文本")
        self.assertEqual(0, result["summary"]["changed_hunks"])
        self.assertEqual(2, result["summary"]["unchanged_lines"])
        self.assertEqual([], result["hunks"])


if __name__ == "__main__":
    unittest.main()
