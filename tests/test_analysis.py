import unittest

from kfault.analysis import (
    build_chains,
    dedupe,
    distribution,
    false_pass,
    format_forward_budget,
    format_warnings,
    forward_budget_audit,
    gpu_label,
    gpu_matches,
    missing_chains,
    outcome,
    record_gpu,
    repair_outcomes,
    select,
)


def record(p, arm, turn, cat, at="2026-07-30T00:00:00", **extra):
    row = {
        "problem_id": p,
        "arm": arm,
        "turn": turn,
        "category": cat,
        "labelled_at": at,
        "level": 2,
        "model_id": "zai.glm-5",
        "classification_config": {"schema": 4, "gpu": "T4"},
    }
    row.update(extra)
    return row


class TestSelect(unittest.TestCase):
    def test_filters_on_every_field(self):
        rows = [
            record(1, "think", 0, "pass"),
            record(2, "think", 0, "pass", model_id="deepseek.v3-v1:0"),
            record(3, "think", 0, "pass", classification_config={"schema": 3}),
        ]
        self.assertEqual(len(select(rows, level=2)), 3)
        self.assertEqual(len(select(rows, model_id="zai.glm-5")), 2)
        self.assertEqual(len(select(rows, schema=4)), 2)
        self.assertEqual(len(select(rows, problems=[1])), 1)
        self.assertEqual(len(select(rows, gpu="T4")), 2)


class TestDedupe(unittest.TestCase):
    def test_keeps_the_most_recent_record(self):
        rows = [
            record(1, "think", 0, "functional", at="2026-07-01"),
            record(1, "think", 0, "pass", at="2026-07-30"),
        ]
        kept = dedupe(rows)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["category"], "pass")

    def test_different_turns_are_not_merged(self):
        rows = [record(1, "think", 0, "syntactic"), record(1, "think", 1, "pass")]
        self.assertEqual(len(dedupe(rows)), 2)


class TestChains(unittest.TestCase):
    def test_ordered_by_turn(self):
        rows = [
            record(1, "think", 2, "pass"),
            record(1, "think", 0, "syntactic"),
            record(1, "think", 1, "functional"),
        ]
        chain = build_chains(rows)[(1, "think")]
        self.assertEqual([a["turn"] for a in chain], [0, 1, 2])


class TestOutcome(unittest.TestCase):
    def test_rescued(self):
        chain = [record(1, "d", 0, "syntactic"), record(1, "d", 1, "pass")]
        self.assertEqual(outcome(chain), "rescued")

    def test_visible_failure_becoming_silent_is_hidden(self):
        chain = [record(1, "d", 0, "syntactic"), record(1, "d", 1, "silent")]
        self.assertEqual(outcome(chain), "hidden")

    def test_still_failing(self):
        chain = [record(1, "d", 0, "functional"), record(1, "d", 1, "functional")]
        self.assertEqual(outcome(chain), "still_failing")

    def test_single_pass_needs_no_repair(self):
        self.assertEqual(outcome([record(1, "d", 0, "pass")]), "no_repair_needed")

    def test_single_visible_failure_means_repair_never_ran(self):
        self.assertEqual(outcome([record(1, "d", 0, "syntactic")]), "no_repair_run")

    def test_counts(self):
        chains = {
            (1, "d"): [record(1, "d", 0, "syntactic"), record(1, "d", 1, "pass")],
            (2, "d"): [record(2, "d", 0, "syntactic"), record(2, "d", 1, "silent")],
            (3, "d"): [record(3, "d", 0, "pass")],
        }
        counts = repair_outcomes(chains)
        self.assertEqual(counts["rescued"], 1)
        self.assertEqual(counts["hidden"], 1)
        self.assertEqual(counts["no_repair_needed"], 1)


class TestDistribution(unittest.TestCase):
    def setUp(self):
        self.chains = {
            (1, "think"): [
                record(1, "think", 0, "syntactic"),
                record(1, "think", 1, "pass"),
            ],
            (2, "think"): [record(2, "think", 0, "silent")],
        }

    def test_baseline_uses_turn_zero(self):
        counts = distribution(self.chains, index=0)["think"]
        self.assertEqual(counts["syntactic"], 1)
        self.assertEqual(counts["pass"], 0)

    def test_final_uses_the_last_turn(self):
        counts = distribution(self.chains, index=-1)["think"]
        self.assertEqual(counts["pass"], 1)
        self.assertEqual(counts["syntactic"], 0)

    def test_false_pass_rate(self):
        rates = false_pass(distribution(self.chains, index=-1))
        self.assertEqual(rates["think"], (1, 2, 50.0))

    def test_false_pass_without_passing_kernels(self):
        chains = {(1, "d"): [record(1, "d", 0, "functional")]}
        self.assertEqual(false_pass(distribution(chains))["d"], (0, 0, None))


class TestWarnings(unittest.TestCase):
    def test_missing_problem_arms_are_reported(self):
        chains = build_chains([record(1, "think", 0, "pass")])
        self.assertEqual(
            missing_chains(chains, [1, 2], ["think", "direct"]),
            [(1, "direct"), (2, "direct"), (2, "think")],
        )

    def test_warning_text_names_the_gap(self):
        rows = [record(1, "think", 0, "pass")]
        chains = build_chains(rows)
        lines = format_warnings(rows, chains, problems=[1, 2], arms=["think", "direct"])
        self.assertTrue(
            any("3 of 4 problem-arms have NO record" in line for line in lines)
        )

    def test_no_warnings_when_complete(self):
        rows = [record(1, "think", 0, "pass"), record(1, "direct", 0, "pass")]
        chains = build_chains(rows)
        self.assertEqual(
            format_warnings(rows, chains, problems=[1], arms=["think", "direct"]), []
        )


class ForwardBudgetAuditTests(unittest.TestCase):
    """Which cached records would the per-forward limit relabel?"""

    def rows(self):
        return [
            # passing kernel, benchmarked, comfortably fast
            {
                "problem_id": 1,
                "arm": "think",
                "category": "pass",
                "timing": {"kernel_ms": 12.0},
            },
            # passing kernel, benchmarked, over a 5s budget
            {
                "problem_id": 2,
                "arm": "direct",
                "category": "pass",
                "timing": {"kernel_ms": 8200.0},
            },
            # silent: classifier skips benchmarking, so no evidence either way
            {"problem_id": 3, "arm": "think", "category": "silent", "timing": None},
            # syntactic: never reached a forward at all
            {"problem_id": 4, "arm": "direct", "category": "syntactic"},
            # a post-fix record carries forward_seconds directly
            {
                "problem_id": 5,
                "arm": "think",
                "category": "pass",
                "forward_seconds": 6.5,
                "timing": {"kernel_ms": 6500.0},
            },
        ]

    def test_buckets(self):
        audit = forward_budget_audit(self.rows(), 5.0)
        self.assertEqual([r["problem_id"] for r in audit["over"]], [2, 5])
        self.assertEqual([r["problem_id"] for r in audit["under"]], [1])
        self.assertEqual([r["problem_id"] for r in audit["unverifiable"]], [3, 4])

    def test_forward_seconds_wins_over_timing(self):
        row = {
            "problem_id": 9,
            "category": "pass",
            "forward_seconds": 0.5,
            "timing": {"kernel_ms": 9999.0},
        }
        audit = forward_budget_audit([row], 5.0)
        self.assertEqual(len(audit["under"]), 1)

    def test_a_larger_budget_clears_the_over_bucket(self):
        audit = forward_budget_audit(self.rows(), 20.0)
        self.assertEqual(audit["over"], [])

    def test_report_names_the_unverifiable_categories(self):
        audit = forward_budget_audit(self.rows(), 5.0)
        text = "\n".join(format_forward_budget(audit, 5.0))
        self.assertIn("relabelled runtime", text)
        self.assertIn("silent", text)
        self.assertIn("syntactic", text)

    def test_all_clear_says_so(self):
        rows = [{"problem_id": 1, "category": "pass", "timing": {"kernel_ms": 5.0}}]
        text = "\n".join(format_forward_budget(forward_budget_audit(rows, 5.0), 5.0))
        self.assertIn("no re-run needed", text)


class RecordGpuTests(unittest.TestCase):
    """classification_config["gpu"] lies; runtime_environment is ground truth."""

    def row(self, cfg_gpu="T4", real=None):
        row = {"level": 2, "problem_id": 1, "classification_config": {"gpu": cfg_gpu}}
        if real is not None:
            row["runtime_environment"] = {"gpu_name": real}
        return row

    def test_prefers_the_real_device_over_the_config(self):
        self.assertEqual(record_gpu(self.row("T4", "NVIDIA L4")), "NVIDIA L4")

    def test_falls_back_to_config_when_no_gpu_was_reached(self):
        # e.g. category "empty": no container ever ran
        self.assertEqual(record_gpu(self.row("T4", None)), "T4")

    def test_short_name_matches_full_device_name(self):
        self.assertTrue(gpu_matches(self.row("T4", "NVIDIA L4"), "L4"))
        self.assertTrue(gpu_matches(self.row("T4", "Tesla T4"), "t4"))

    def test_mislabelled_l4_record_is_not_returned_as_t4(self):
        # the exact 2026-08-03 bug: config says T4, hardware was L4
        self.assertFalse(gpu_matches(self.row("T4", "NVIDIA L4"), "T4"))

    def test_select_filters_on_the_real_device(self):
        rows = [self.row("T4", "Tesla T4"), self.row("T4", "NVIDIA L4")]
        self.assertEqual(len(select(rows, gpu="L4")), 1)
        self.assertEqual(len(select(rows, gpu="T4")), 1)
        self.assertEqual(len(select(rows, gpu=None)), 2)

    def test_unknown_device_never_matches_a_filter(self):
        self.assertFalse(gpu_matches({"level": 2}, "T4"))


if __name__ == "__main__":
    unittest.main()


class GpuLabelTests(unittest.TestCase):
    """One device must not appear as two rows in the report."""

    def test_full_device_names_collapse_to_a_token(self):
        def label(name):
            return gpu_label({"runtime_environment": {"gpu_name": name}})

        self.assertEqual(label("NVIDIA L4"), "L4")
        self.assertEqual(label("Tesla T4"), "T4")

    def test_config_fallback_gives_the_same_token(self):
        # a timed-out container never reported a device, so only the config
        # value ("L4") survives; it must group with the "NVIDIA L4" records
        self.assertEqual(gpu_label({"classification_config": {"gpu": "L4"}}), "L4")

    def test_one_device_yields_one_group(self):
        rows = [
            {"runtime_environment": {"gpu_name": "NVIDIA L4"}},
            {"classification_config": {"gpu": "L4"}},
        ]
        self.assertEqual({gpu_label(r) for r in rows}, {"L4"})

    def test_distinct_devices_stay_distinct(self):
        rows = [
            {"runtime_environment": {"gpu_name": "NVIDIA L4"}},
            {"runtime_environment": {"gpu_name": "Tesla T4"}},
        ]
        self.assertEqual({gpu_label(r) for r in rows}, {"L4", "T4"})

    def test_unknown_device_is_none(self):
        self.assertIsNone(gpu_label({}))

    def test_unrecognised_name_is_passed_through(self):
        row = {"runtime_environment": {"gpu_name": "Some Future GPU"}}
        self.assertEqual(gpu_label(row), "Some Future GPU")
