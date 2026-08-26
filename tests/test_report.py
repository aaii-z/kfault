import unittest
from collections import Counter

from kfault.report import category_columns, format_false_pass, format_rows


class ReportTests(unittest.TestCase):
    def test_console_summary_formatting(self):
        # canonical category order first, unexpected labels appended
        self.assertEqual(
            category_columns({"think": Counter({"pass": 1, "empty": 2, "error": 1})}),
            ["syntactic", "runtime", "functional", "silent", "pass", "empty", "error"],
        )

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
        # a missing detail must not crash the summary
        self.assertIn(
            "error",
            format_rows(1, [(1, "think", 0, {"category": "error", "detail": None})])[0],
        )

        per_arm = {
            "think": Counter({"pass": 3, "silent": 1}),
            "direct": Counter({"functional": 2}),
        }
        lines = format_false_pass(("think", "direct"), per_arm)
        self.assertIn("1/4 = 25%", lines[0])
        self.assertIn("no standard-passing kernels", lines[1])


if __name__ == "__main__":
    unittest.main()
