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


def gpu_row(cfg_gpu="T4", real=None):
    row = {"level": 2, "problem_id": 1, "classification_config": {"gpu": cfg_gpu}}
    if real is not None:
        row["runtime_environment"] = {"gpu_name": real}
    return row


BUDGET_ROWS = [
    {
        "problem_id": 1,
        "arm": "think",
        "category": "pass",
        "timing": {"kernel_ms": 12.0},
    },
    {
        "problem_id": 2,
        "arm": "direct",
        "category": "pass",
        "timing": {"kernel_ms": 8200.0},
    },
    # silent: the classifier skips benchmarking, so there is no evidence either way
    {"problem_id": 3, "arm": "think", "category": "silent", "timing": None},
    # syntactic: never reached a forward pass at all
    {"problem_id": 4, "arm": "direct", "category": "syntactic"},
    {
        "problem_id": 5,
        "arm": "think",
        "category": "pass",
        "forward_seconds": 6.5,
        "timing": {"kernel_ms": 6500.0},
    },
]


class SelectionTests(unittest.TestCase):
    def test_rows_are_filtered_and_deduplicated(self):
        rows = [
            record(1, "think", 0, "pass"),
            record(2, "think", 0, "pass", model_id="deepseek.v3-v1:0"),
            record(3, "think", 0, "pass", classification_config={"schema": 3}),
        ]
        for kwargs, expected in (
            ({"level": 2}, 3),
            ({"model_id": "zai.glm-5"}, 2),
            ({"schema": 4}, 2),
            ({"problems": [1]}, 1),
            ({"gpu": "T4"}, 2),
        ):
            with self.subTest(**kwargs):
                self.assertEqual(len(select(rows, **kwargs)), expected)

        # a re-run leaves an older copy behind; the newest record wins
        kept = dedupe(
            [
                record(1, "think", 0, "functional", at="2026-07-01"),
                record(1, "think", 0, "pass", at="2026-07-30"),
            ]
        )
        self.assertEqual([r["category"] for r in kept], ["pass"])
        # but two turns of one chain are distinct records, not duplicates
        self.assertEqual(
            len(
                dedupe(
                    [record(1, "think", 0, "syntactic"), record(1, "think", 1, "pass")]
                )
            ),
            2,
        )

    def test_chains_and_outcomes(self):
        rows = [
            record(1, "think", 2, "pass"),
            record(1, "think", 0, "syntactic"),
            record(1, "think", 1, "functional"),
        ]
        self.assertEqual(
            [a["turn"] for a in build_chains(rows)[(1, "think")]], [0, 1, 2]
        )

        cases = (
            ("rescued", [record(1, "d", 0, "syntactic"), record(1, "d", 1, "pass")]),
            # the effect this study measures: a visible failure becomes invisible
            ("hidden", [record(1, "d", 0, "syntactic"), record(1, "d", 1, "silent")]),
            (
                "still_failing",
                [record(1, "d", 0, "functional"), record(1, "d", 1, "functional")],
            ),
            ("no_repair_needed", [record(1, "d", 0, "pass")]),
            ("no_repair_run", [record(1, "d", 0, "syntactic")]),
        )
        for expected, chain in cases:
            with self.subTest(outcome=expected):
                self.assertEqual(outcome(chain), expected)

        counts = repair_outcomes({key: chain for key, (_, chain) in enumerate(cases)})
        self.assertEqual(counts["rescued"], 1)
        self.assertEqual(counts["hidden"], 1)

    def _rate_chains(self):
        return {
            (1, "think"): [
                record(1, "think", 0, "syntactic"),
                record(1, "think", 1, "pass"),
            ],
            (2, "think"): [record(2, "think", 0, "silent")],
        }

    def test_rates_and_completeness(self):
        self.chains = self._rate_chains()
        first = distribution(self.chains, index=0)["think"]
        self.assertEqual((first["syntactic"], first["pass"]), (1, 0))
        last = distribution(self.chains, index=-1)["think"]
        self.assertEqual((last["pass"], last["syntactic"]), (1, 0))
        # false pass is silent over silent plus pass, undefined when nothing passed
        self.assertEqual(
            false_pass(distribution(self.chains, index=-1))["think"], (1, 2, 50.0)
        )
        empty = {(1, "d"): [record(1, "d", 0, "functional")]}
        self.assertEqual(false_pass(distribution(empty))["d"], (0, 0, None))

        rows = [record(1, "think", 0, "pass")]
        chains = build_chains(rows)
        self.assertEqual(
            missing_chains(chains, [1, 2], ["think", "direct"]),
            [(1, "direct"), (2, "direct"), (2, "think")],
        )
        lines = format_warnings(rows, chains, problems=[1, 2], arms=["think", "direct"])
        self.assertTrue(any("3 of 4 problem-arms have NO record" in x for x in lines))

        complete = [record(1, "think", 0, "pass"), record(1, "direct", 0, "pass")]
        self.assertEqual(
            format_warnings(
                complete, build_chains(complete), problems=[1], arms=["think", "direct"]
            ),
            [],
        )


class AuditTests(unittest.TestCase):
    def test_forward_limit_buckets_and_device_identity(self):
        audit = forward_budget_audit(BUDGET_ROWS, 5.0)
        self.assertEqual([r["problem_id"] for r in audit["over"]], [2, 5])
        self.assertEqual([r["problem_id"] for r in audit["under"]], [1])
        self.assertEqual([r["problem_id"] for r in audit["unverifiable"]], [3, 4])
        # a measured forward time is trusted over an inferred one
        measured = {
            "problem_id": 9,
            "category": "pass",
            "forward_seconds": 0.5,
            "timing": {"kernel_ms": 9999.0},
        }
        self.assertEqual(len(forward_budget_audit([measured], 5.0)["under"]), 1)
        self.assertEqual(forward_budget_audit(BUDGET_ROWS, 20.0)["over"], [])

        text = "\n".join(
            format_forward_budget(forward_budget_audit(BUDGET_ROWS, 5.0), 5.0)
        )
        for fragment in ("relabelled runtime", "silent", "syntactic"):
            self.assertIn(fragment, text)
        clear = [{"problem_id": 1, "category": "pass", "timing": {"kernel_ms": 5.0}}]
        self.assertIn(
            "no re-run needed",
            "\n".join(format_forward_budget(forward_budget_audit(clear, 5.0), 5.0)),
        )

        # the configured device lies; the device the kernel ran on is the truth
        self.assertEqual(record_gpu(gpu_row("T4", "NVIDIA L4")), "NVIDIA L4")
        # no container ever ran, so only the configured value survives
        self.assertEqual(record_gpu(gpu_row("T4", None)), "T4")
        self.assertTrue(gpu_matches(gpu_row("T4", "NVIDIA L4"), "L4"))
        self.assertTrue(gpu_matches(gpu_row("T4", "Tesla T4"), "t4"))
        # the 2026-08-03 defect: config said T4, the hardware was an L4
        self.assertFalse(gpu_matches(gpu_row("T4", "NVIDIA L4"), "T4"))
        self.assertFalse(gpu_matches({"level": 2}, "T4"))

        rows = [gpu_row("T4", "Tesla T4"), gpu_row("T4", "NVIDIA L4")]
        self.assertEqual(len(select(rows, gpu="L4")), 1)
        self.assertEqual(len(select(rows, gpu=None)), 2)

        for name, expected in (
            ("NVIDIA L4", "L4"),
            ("Tesla T4", "T4"),
            ("Some Future GPU", "Some Future GPU"),
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    gpu_label({"runtime_environment": {"gpu_name": name}}), expected
                )
        # a timed-out container reported no device, so it must still group with
        # the records that did
        self.assertEqual(gpu_label({"classification_config": {"gpu": "L4"}}), "L4")
        self.assertIsNone(gpu_label({}))


if __name__ == "__main__":
    unittest.main()
