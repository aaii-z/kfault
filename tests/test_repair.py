import unittest

from kfault.repair import build_feedback, build_repair_prompt, should_repair, tail

TIMEOUT = {
    "failure": {
        "stage": "container_timeout",
        "error_type": "FunctionTimeoutError",
        "message": (
            "Task's current input in-01KZ2ECPY2KFW2A1M67QHXHGBH:1785 "
            "hit its timeout of 900s"
        ),
    }
}


class RepairPolicyTests(unittest.TestCase):
    def test_repair_policy_and_what_is_withheld(self):
        for category in ("syntactic", "runtime", "functional", "empty"):
            with self.subTest(category=category):
                self.assertTrue(should_repair(category))
        # pass ends the loop; silent must never reach the model at all
        for category in ("pass", "silent", "error"):
            with self.subTest(category=category):
                self.assertFalse(should_repair(category))

        battery = {"battery_diagnostics": {"non_contig": {"status": "mismatch"}}}
        self.assertEqual(build_feedback("silent", "non_contig", battery), "")
        self.assertEqual(build_feedback("pass", "survived core battery", {}), "")

        # nvcc puts the decisive line last, so the tail is what matters
        result = tail("head" + "x" * 1000 + "decisive line", limit=50)
        self.assertTrue(result.endswith("decisive line"))
        self.assertNotIn("head", result)
        self.assertEqual(tail("boom"), "boom")
        self.assertEqual(tail(None), "")


class FeedbackContentTests(unittest.TestCase):
    def test_feedback_content(self):
        build = build_feedback(
            "syntactic",
            "error: no type floatt",
            {"failure": {"stage": "build", "message": "error: no type floatt"}},
        )
        self.assertIn("syntactic", build)
        self.assertIn("no type floatt", build)

        crash = build_feedback(
            "runtime",
            "illegal access",
            {
                "failure": {"stage": "standard_forward", "message": "illegal access"},
                "sanitizer": {"log_excerpt": "Invalid __global__ write of size 4"},
            },
        )
        self.assertIn("Invalid __global__ write", crash)

        slow = build_feedback(
            "runtime",
            "slow",
            {
                "failure": {
                    "stage": "forward_time_limit",
                    "message": "one warm forward took 21.4s, over the 5.0s limit",
                }
            },
        )
        self.assertIn("21.4s", slow)

        text = build_feedback("runtime", "timeout", TIMEOUT)
        self.assertNotIn("Task's current input", text)
        self.assertNotIn("in-01KZ", text)
        self.assertIn("did not finish", text)
        for hint in ("loop", "thread"):
            self.assertIn(hint, text.lower())

        prompt = build_repair_prompt(
            "class Model: pass", "class ModelNew: pass", "Build error: boom"
        )
        for fragment in (
            "class Model: pass",
            "class ModelNew: pass",
            "Build error: boom",
        ):
            self.assertIn(fragment, prompt)


if __name__ == "__main__":
    unittest.main()
