import unittest

from kfault.repair import (
    build_feedback,
    build_repair_prompt,
    should_repair,
    tail,
)


class TestShouldRepair(unittest.TestCase):
    def test_visible_failures_are_repairable(self):
        for category in ("syntactic", "runtime", "functional", "empty"):
            self.assertTrue(should_repair(category), category)

    def test_pass_and_hidden_failures_stop_the_loop(self):
        for category in ("pass", "silent", "error"):
            self.assertFalse(should_repair(category), category)


class TestTail(unittest.TestCase):
    def test_short_text_is_untouched(self):
        self.assertEqual(tail("boom"), "boom")

    def test_long_text_keeps_the_end(self):
        text = "head" + "x" * 1000 + "decisive line"
        result = tail(text, limit=50)
        self.assertTrue(result.endswith("decisive line"))
        self.assertNotIn("head", result)

    def test_none_is_empty(self):
        self.assertEqual(tail(None), "")


class TestBuildFeedback(unittest.TestCase):
    def test_syntactic_reports_the_build_error(self):
        extra = {"failure": {"stage": "build", "message": "error: no type floatt"}}
        feedback = build_feedback("syntactic", "error: no type floatt", extra)
        self.assertIn("syntactic", feedback)
        self.assertIn("build", feedback)
        self.assertIn("no type floatt", feedback)

    def test_runtime_includes_the_sanitizer_excerpt(self):
        extra = {
            "failure": {"stage": "standard_forward", "message": "illegal access"},
            "sanitizer": {"log_excerpt": "Invalid __global__ write of size 4"},
        }
        feedback = build_feedback("runtime", "illegal access", extra)
        self.assertIn("Invalid __global__ write", feedback)

    def test_silent_leaks_nothing(self):
        extra = {"battery_diagnostics": {"non_contig": {"status": "mismatch"}}}
        self.assertEqual(build_feedback("silent", "non_contig", extra), "")

    def test_pass_leaks_nothing(self):
        self.assertEqual(build_feedback("pass", "survived core battery", {}), "")


class TestBuildRepairPrompt(unittest.TestCase):
    def test_contains_problem_attempt_and_error(self):
        prompt = build_repair_prompt(
            "class Model: pass", "class ModelNew: pass", "Build error: boom"
        )
        self.assertIn("class Model: pass", prompt)
        self.assertIn("class ModelNew: pass", prompt)
        self.assertIn("Build error: boom", prompt)


if __name__ == "__main__":
    unittest.main()


class TimeoutFeedbackTests(unittest.TestCase):
    """A timeout must describe the kernel, not quote the harness."""

    def timeout_extra(self):
        return {
            "failure": {
                "stage": "container_timeout",
                "error_type": "FunctionTimeoutError",
                "message": (
                    "Task's current input in-01KZ2ECPY2KFW2A1M67QHXHGBH:1785 "
                    "hit its timeout of 900s"
                ),
            }
        }

    def test_timeout_feedback_does_not_leak_the_harness_message(self):
        text = build_feedback("runtime", "…timeout…", self.timeout_extra())
        self.assertNotIn("Task's current input", text)
        self.assertNotIn("in-01KZ", text)

    def test_timeout_feedback_is_actionable(self):
        text = build_feedback("runtime", "…timeout…", self.timeout_extra())
        self.assertIn("did not finish", text)
        for hint in ("loop", "thread"):
            self.assertIn(hint, text.lower())

    def test_slow_forward_feedback_keeps_the_measured_detail(self):
        extra = {
            "failure": {
                "stage": "forward_time_limit",
                "message": "one warm forward took 21.4s, over the 5.0s budget",
            }
        }
        text = build_feedback("runtime", "slow", extra)
        self.assertIn("21.4s", text)
        self.assertIn("too", text.lower())

    def test_a_normal_runtime_error_still_shows_the_real_error(self):
        extra = {
            "failure": {
                "stage": "standard_forward",
                "message": "CUDA error: an illegal memory access was encountered",
            }
        }
        text = build_feedback("runtime", "crash", extra)
        self.assertIn("illegal memory access", text)

    def test_timeout_is_still_repairable(self):
        self.assertTrue(should_repair("runtime"))
