"""Does the strict/extended definition change how the models rank?

EvalPlus (Liu et al., NeurIPS 2023) found that insufficient tests do not only
depress pass rates, they reorder models. This checks the same question on
`data/attempts.csv` and `data/chains.csv`.

strict   -- a kernel counts as passing if it satisfies KernelBench's own
            contract: repeated random *contiguous* draws of the declared
            shape, plus determinism. Only these probes may condemn it.
extended -- drop-in-replacement semantics: any battery probe may condemn it.
            This is the labelling used everywhere else in the project.

The two definitions differ only in the relabelling of `silent` attempts, and a
`silent` attempt ends its chain exactly as a `pass` does (stage 4 is withheld
from the repair prompt), so chain structure is identical under both. No
counterfactual re-run is needed.

    uv run python misranking.py
"""

import collections
import csv
from math import comb

from kfault.config import CLASSIFIER_CACHE_VERSION

FAIL = {"mismatch", "kernel_error"}

# Probes inside KernelBench's stated contract. Everything else in the battery
# is an extension of it.
STRICT_PROBES = (
    "probe_fresh_seed_1",
    "probe_fresh_seed_2",
    "probe_fresh_seed_3",
    "probe_nondeterministic",
)

SHORT = {
    "zai.glm-5": "GLM-5",
    "deepseek.v3-v1:0": "DeepSeek V3.2",
    "qwen.qwen3-235b-a22b-2507-v1:0": "Qwen3-235B",
}


def load():
    attempts = list(csv.DictReader(open("data/attempts.csv")))
    chains = list(csv.DictReader(open("data/chains.csv")))
    # A superseded classifier is not comparable with the current one: schema 3
    # predates the clone_tree stride fix, so its non_contig probe tested a
    # contiguous tensor and could never report a silent failure.
    schemas = {row["schema"] for row in attempts}
    if schemas - {str(CLASSIFIER_CACHE_VERSION)}:
        raise SystemExit(
            f"data/attempts.csv holds schema {sorted(schemas)}, expected only "
            f"{CLASSIFIER_CACHE_VERSION}. Re-run export.py."
        )
    by_chain = collections.defaultdict(list)
    for row in attempts:
        key = (row["model_id"], row["device"], row["problem_id"], row["arm"])
        by_chain[key].append(row)
    for rows in by_chain.values():
        rows.sort(key=lambda r: int(r["turn"]))
    return chains, by_chain


def outcomes(chains, by_chain, arm=None, turn="final"):
    """(model, device, problem) -> (strict_pass, extended_pass)."""
    result = {}
    for chain in chains:
        if arm is not None and chain["arm"] != arm:
            continue
        key = (chain["model_id"], chain["device"], chain["problem_id"], chain["arm"])
        rows = by_chain[key]
        row = rows[0] if turn == "oneshot" else rows[-1]
        extended = row["category"] == "pass"
        violates = any(row[p] in FAIL for p in STRICT_PROBES)
        strict = extended or (row["category"] == "silent" and not violates)
        result[(SHORT[chain["model_id"]], chain["device"], chain["problem_id"])] = (
            strict,
            extended,
        )
    return result


def rates(result, device=None):
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for (model, dev, _), (strict, extended) in result.items():
        if device is not None and dev != device:
            continue
        row = agg[model]
        row[0] += 1
        row[1] += strict
        row[2] += extended
    return agg


def show(agg, label):
    print(f"-- {label}")
    print(f"   {'model':12} {'n':>4} {'strict':>8} {'extended':>10} {'gap':>6}")
    for model, (n, s, e) in sorted(agg.items(), key=lambda kv: -kv[1][1] / kv[1][0]):
        print(
            f"   {model:12} {n:4} {100 * s / n:7.1f}% {100 * e / n:9.1f}%"
            f" {100 * (s - e) / n:5.1f}"
        )
    order = lambda i: " > ".join(  # noqa: E731
        m for m, _ in sorted(agg.items(), key=lambda kv: -kv[1][i] / kv[1][0])
    )
    print(f"   strict rank  : {order(1)}")
    print(f"   extended rank: {order(2)}")
    print(f"   RANKING CHANGES: {order(1) != order(2)}")
    print()


def two_sided_binomial(successes, trials):
    """Exact two-sided binomial test against p=0.5, no scipy needed.

    Symmetric under p=0.5, so the two-sided p is twice the smaller tail,
    capped at 1.
    """
    if trials == 0:
        return 1.0
    tail = min(successes, trials - successes)
    cumulative = sum(comb(trials, k) for k in range(tail + 1)) / 2**trials
    return min(1.0, 2 * cumulative)


def mcnemar(result, left, right, index, device=None):
    """Paired sign test over problems solved by exactly one of the two."""
    only_left = only_right = 0
    for (model, dev, problem), verdict in result.items():
        if model != left or (device is not None and dev != device):
            continue
        other = result.get((right, dev, problem))
        if other is None:
            continue
        if verdict[index] and not other[index]:
            only_left += 1
        elif other[index] and not verdict[index]:
            only_right += 1
    n = only_left + only_right
    return only_left, only_right, two_sided_binomial(only_left, n)


def main():
    chains, by_chain = load()

    for arm in (None, "think", "direct"):
        label = arm or "both arms"
        show(rates(outcomes(chains, by_chain, arm)), f"{label}, after repair")

    show(
        rates(outcomes(chains, by_chain, "think", turn="oneshot")),
        "think, one-shot (turn 0)",
    )

    result = outcomes(chains, by_chain, "think")
    models = sorted({m for m, _, _ in result})
    for name, index in (("strict", 0), ("extended", 1)):
        print(f"-- McNemar, think arm, after repair, {name}")
        for i, left in enumerate(models):
            for right in models[i + 1 :]:
                a, b, p = mcnemar(result, left, right, index)
                mark = " *" if p < 0.05 else ""
                print(f"   pooled  {left:12} {a:3} / {b:<3} {right:12} p={p:.4f}{mark}")
                for device in ("L4", "T4"):
                    a, b, p = mcnemar(result, left, right, index, device)
                    print(
                        f"     {device}    {left:12} {a:3} / {b:<3}"
                        f" {right:12} p={p:.4f}"
                    )
        print()
    print("Pooled tests treat L4 and T4 as independent, which they are not:")
    print("the same problem and model appears twice. Per-device rows are the")
    print("conservative reading.")


if __name__ == "__main__":
    main()
