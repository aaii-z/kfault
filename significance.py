"""Statistical tests for the relocation claim, from the exported tables.

    uv run python significance.py

One greedy sample per cell means no repeated measurement of the *same* problem,
so a rate cannot be given a sampling-error bar in the usual way. Two things are
still testable, and both use only data already collected.

1. Repair-produced acceptances versus first-attempt acceptances. If repair
   produced kernels of the same quality as first attempts, the share that are
   silent would be the same in both groups. Fisher's exact test compares them.
   This is the sharpest available test of the relocation claim, because it
   contrasts two disjoint sets of kernels rather than two rates over shifting
   denominators.

2. A bootstrap interval over problems. The 100 Level 2 problems are one sample
   from the space of problems the benchmark could have contained, so resampling
   them with replacement gives an interval for the false-pass rate that
   reflects which problems were chosen.

Neither addresses generation variance, which needs several samples per problem.
That remains open and is stated as such in the dissertation.
"""

import collections
import csv
import random
from math import comb

from kfault.config import CLASSIFIER_CACHE_VERSION

ACCEPTED = {"pass", "silent"}  # what KernelBench would call a success

SHORT = {
    "zai.glm-5": "GLM-5",
    "deepseek.v3-v1:0": "DeepSeek V3.2",
    "qwen.qwen3-235b-a22b-2507-v1:0": "Qwen3-235B",
}


def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact test on [[a, b], [c, d]], no scipy needed.

    Sums the hypergeometric probability of every table at least as extreme as
    the observed one. Verified against scipy.stats.fisher_exact.
    """
    total = a + b + c + d
    if min(a + b, c + d, a + c, b + d) < 0 or total == 0:
        return 1.0

    def probability(x):
        return comb(a + b, x) * comb(c + d, a + c - x) / comb(total, a + c)

    low, high = max(0, a + c - (c + d)), min(a + b, a + c)
    observed = probability(a)
    return min(
        1.0,
        sum(
            probability(x)
            for x in range(low, high + 1)
            if probability(x) <= observed * (1 + 1e-9)
        ),
    )


def load():
    attempts = list(csv.DictReader(open("data/attempts.csv")))
    chains = list(csv.DictReader(open("data/chains.csv")))
    schemas = {row["schema"] for row in attempts}
    if schemas - {str(CLASSIFIER_CACHE_VERSION)}:
        raise SystemExit(
            f"data/attempts.csv holds schema {sorted(schemas)}, expected only "
            f"{CLASSIFIER_CACHE_VERSION}. Re-run export.py."
        )
    by_chain = collections.defaultdict(list)
    for row in attempts:
        by_chain[
            (row["model_id"], row["device"], row["problem_id"], row["arm"])
        ].append(row)
    for rows in by_chain.values():
        rows.sort(key=lambda r: int(r["turn"]))
    return chains, by_chain


def split(chains, by_chain, keep):
    """(first-attempt silent, n, repair-produced silent, n).

    A chain lands in the first group if its one-shot kernel was already
    accepted, and in the second only if repair is what made it acceptable.
    Chains that never reach an accepted kernel appear in neither.
    """
    fs = fn = rs = rn = 0
    for chain in chains:
        if not keep(chain):
            continue
        rows = by_chain[
            (chain["model_id"], chain["device"], chain["problem_id"], chain["arm"])
        ]
        first, last = rows[0]["category"], rows[-1]["category"]
        if first in ACCEPTED:
            fn += 1
            fs += first == "silent"
        elif last in ACCEPTED:
            rn += 1
            rs += last == "silent"
    return fs, fn, rs, rn


def report(chains, by_chain, keep, label):
    fs, fn, rs, rn = split(chains, by_chain, keep)
    if not fn or not rn:
        return
    p = fisher_exact(fs, fn - fs, rs, rn - rs)
    mark = " *" if p < 0.05 else ""
    print(
        f"  {label:22} first {fs:3}/{fn:<4}={100 * fs / fn:5.1f}%   "
        f"repaired {rs:3}/{rn:<4}={100 * rs / rn:5.1f}%   p={p:.4f}{mark}"
    )


def bootstrap(chains, by_chain, model, arm, which, draws=20000, seed=7):
    """Percentile interval for the false-pass rate, resampling problems."""
    rnd = random.Random(seed)
    by_problem = collections.defaultdict(list)
    for chain in chains:
        if chain["model_id"] != model or chain["arm"] != arm:
            continue
        rows = by_chain[
            (chain["model_id"], chain["device"], chain["problem_id"], chain["arm"])
        ]
        by_problem[chain["problem_id"]].append(rows[0] if which == "one" else rows[-1])
    problems = sorted(by_problem)
    rates = []
    for _ in range(draws):
        silent = total = 0
        for _ in problems:
            for row in by_problem[rnd.choice(problems)]:
                if row["category"] == "silent":
                    silent += 1
                    total += 1
                elif row["category"] == "pass":
                    total += 1
        if total:
            rates.append(100 * silent / total)
    rates.sort()
    return rates[int(0.025 * len(rates))], rates[int(0.975 * len(rates))]


def rate(chains, by_chain, model, arm, which):
    counts = collections.Counter()
    for chain in chains:
        if chain["model_id"] != model or chain["arm"] != arm:
            continue
        rows = by_chain[
            (chain["model_id"], chain["device"], chain["problem_id"], chain["arm"])
        ]
        counts[(rows[0] if which == "one" else rows[-1])["category"]] += 1
    silent, passed = counts["silent"], counts["pass"]
    return 100 * silent / (silent + passed) if silent + passed else 0.0


def main():
    chains, by_chain = load()

    print("Are the kernels repair produces more often silent than first attempts?")
    print("Fisher exact, two-sided.\n")
    report(chains, by_chain, lambda c: c["arm"] == "think", "think, all")
    report(chains, by_chain, lambda c: c["arm"] == "direct", "direct, all")
    print()
    for model, name in SHORT.items():
        report(
            chains,
            by_chain,
            lambda c, m=model: c["model_id"] == m and c["arm"] == "think",
            f"{name}, think",
        )
    print()
    # Pooling both GPUs counts every problem twice. Each device alone does not.
    for device in ("L4", "T4"):
        report(
            chains,
            by_chain,
            lambda c, d=device: c["device"] == d and c["arm"] == "think",
            f"{device} only, think",
        )

    print("\n\nBootstrap 95% interval for the false-pass rate, resampling the")
    print("100 problems with replacement. Reasoning arm.\n")
    print(f"  {'model':14}{'one-shot':>24}{'after repair':>24}")
    for model, name in SHORT.items():
        lo0, hi0 = bootstrap(chains, by_chain, model, "think", "one")
        lo1, hi1 = bootstrap(chains, by_chain, model, "think", "fin")
        r0 = rate(chains, by_chain, model, "think", "one")
        r1 = rate(chains, by_chain, model, "think", "fin")
        print(
            f"  {name:14}{r0:8.1f}% [{lo0:4.1f}, {hi0:4.1f}]"
            f"{r1:11.1f}% [{lo1:4.1f}, {hi1:4.1f}]"
        )
    print("\n  The per-model intervals overlap, so the rise within any single")
    print("  model is not established on its own. The Fisher tests above")
    print("  compare disjoint sets of kernels and are the stronger evidence.")


if __name__ == "__main__":
    main()
