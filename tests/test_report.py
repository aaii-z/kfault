import unittest
from collections import Counter

from kfault.report import category_columns, format_false_pass, format_rows

ARMS = ("think", "direct")


class TestCategoryColumns(unittest.TestCase):
    def test_canonical_order_then_extras(self):
        per_arm = {"think": Counter({"pass": 1, "empty": 2, "error": 1})}
        self.assertEqual(
            category_columns(per_arm),
            ["syntactic", "runtime", "functional", "silent", "pass", "empty", "error"],
        )


class TestFormatRows(unittest.TestCase):
    def test_marks_cached_and_sanitizer(self):
        rows = [
            (1, "think", 0, {"category": "pass", "detail": "ok", "cached": True}),
            (
                2,
                "direct",
                0,
                {
                    "category": "runtime",
                    "detail": "boom",
                    "sanitizer": {"status": "violation"},
                },
            ),
        ]
        lines = format_rows(2, rows)
        self.assertIn("(cached)", lines[0])
        self.assertIn("san=violation", lines[1])

    def test_missing_detail_does_not_crash(self):
        rows = [(1, "think", 0, {"category": "error", "detail": None})]
        self.assertIn("error", format_rows(1, rows)[0])


class TestFalsePass(unittest.TestCase):
    def test_rate_over_standard_passing_kernels(self):
        per_arm = {
            "think": Counter({"pass": 3, "silent": 1}),
            "direct": Counter({"functional": 2}),
        }
        lines = format_false_pass(ARMS, per_arm)
        self.assertIn("1/4 = 25%", lines[0])
        self.assertIn("no standard-passing kernels", lines[1])


if __name__ == "__main__":
    unittest.main()
