"""Turn the classification cache into per-batch tables.

Pure: takes plain dicts, returns plain data and text. No Modal, no torch, so
it is testable locally.
"""

from collections import Counter

from kfault.report import category_columns

VISIBLE = {"syntactic", "runtime", "functional", "empty"}


def record_gpu(row):
    """The device that actually ran this record.

    ``classification_config["gpu"]`` is unreliable for records written before
    KFAULT_GPU was forwarded into the image. ``runtime_environment.gpu_name`` is
    written on the device itself. Fall back to the config only for records that
    never reached a GPU, such as category "empty".
    """
    name = (row.get("runtime_environment") or {}).get("gpu_name")
    if name:
        return name
    return (row.get("classification_config") or {}).get("gpu")


def gpu_label(row):
    """Canonical short device name ("L4", "T4") for grouping and warnings.

    The device reports "NVIDIA L4"; the config fallback holds bare "L4".
    Grouping on the raw string splits one device into two rows.
    """
    name = record_gpu(row)
    if not name:
        return None
    words = name.upper().replace("-", " ").split()
    for token in ("H100", "A100", "L40S", "A10G", "V100", "L4", "T4"):
        if token in words:
            return token
    return name


def gpu_matches(row, wanted):
    """Match a short name ("L4") against a full device name ("NVIDIA L4")."""
    actual = record_gpu(row)
    return bool(actual) and wanted.strip().lower() in actual.lower()


def select(rows, *, level=None, problems=None, model_id=None, schema=None, gpu=None):
    """Filter raw classification records."""
    chosen = []
    for row in rows:
        config = row.get("classification_config") or {}
        if level is not None and row.get("level") != level:
            continue
        if problems and row.get("problem_id") not in problems:
            continue
        if model_id and row.get("model_id") != model_id:
            continue
        if schema is not None and config.get("schema") != schema:
            continue
        if gpu and not gpu_matches(row, gpu):
            continue
        chosen.append(row)
    return chosen


def dedupe(rows):
    """Keep the most recent record per (problem, arm, turn).

    Re-runs and older sessions leave several turn-0 records for the same
    problem-arm. Without this the same attempt is counted more than once.
    """
    best = {}
    for row in rows:
        key = (row.get("problem_id"), row.get("arm"), row.get("turn") or 0)
        current = best.get(key)
        if current is None or (row.get("labelled_at") or "") > (
            current.get("labelled_at") or ""
        ):
            best[key] = row
    return list(best.values())


def build_chains(rows):
    """Group into ``{(problem, arm): [attempt, ...]}`` ordered by turn."""
    chains = {}
    for row in rows:
        key = (row.get("problem_id"), row.get("arm"))
        chains.setdefault(key, []).append(row)
    for attempts in chains.values():
        attempts.sort(key=lambda row: row.get("turn") or 0)
    return dict(sorted(chains.items()))


def outcome(chain):
    """Classify one repair chain."""
    first = chain[0]["category"]
    last = chain[-1]["category"]
    if len(chain) == 1:
        return "no_repair_needed" if first not in VISIBLE else "no_repair_run"
    if last == "pass":
        return "rescued"
    if last == "silent":
        # only "hidden" if repair made a visible failure invisible
        return "hidden" if first in VISIBLE else "silent_throughout"
    if last in VISIBLE:
        return "still_failing"
    return "other"


def false_pass_rate(counts):
    """(silent, total, percent) from a category counter; percent None if empty."""
    silent, passed = counts.get("silent", 0), counts.get("pass", 0)
    total = silent + passed
    return silent, total, (100 * silent / total if total else None)


def repair_outcomes(chains):
    return Counter(outcome(chain) for chain in chains.values())


def distribution(chains, *, index=-1):
    """Category counts per arm. index=0 is the one-shot baseline, -1 the final."""
    per_arm = {}
    for (_, arm), chain in chains.items():
        per_arm.setdefault(arm, Counter())[chain[index]["category"]] += 1
    return per_arm


def false_pass(per_arm):
    """silent / (silent + pass) per arm, or None when nothing passed."""
    return {arm: false_pass_rate(counts) for arm, counts in per_arm.items()}


def forward_budget_audit(rows, limit_s):
    """Split records by whether the per-forward time limit would relabel them.

    Records cached before the limit carry no ``forward_seconds``; for a passing
    kernel ``timing.kernel_ms`` is the same quantity. Other categories skip
    benchmarking entirely, so they are reported as unverifiable rather than
    assumed fast.

    Returns ``{"over": [...], "under": [...], "unverifiable": [...]}``.
    """
    limit_ms = limit_s * 1000.0
    result = {"over": [], "under": [], "unverifiable": []}
    for row in rows:
        if row.get("forward_seconds") is not None:
            observed = row["forward_seconds"] * 1000.0
        else:
            timing = row.get("timing") or {}
            observed = timing.get("kernel_ms")
        if observed is None:
            result["unverifiable"].append(row)
            continue
        bucket = "over" if observed > limit_ms else "under"
        result[bucket].append({**row, "observed_ms": observed})
    return result


def format_forward_budget(audit, limit_s):
    """Human-readable verdict for forward_budget_audit."""
    over, under = audit["over"], audit["unverifiable"]
    lines = [
        f"budget {limit_s:.1f}s per forward "
        f"({len(audit['under'])} verified under, {len(audit['over'])} over, "
        f"{len(audit['unverifiable'])} unverifiable)"
    ]
    if audit["over"]:
        lines.append("")
        lines.append("would be relabelled runtime on a fresh run:")
        for row in sorted(over, key=lambda r: -r["observed_ms"]):
            lines.append(
                f"  p{row.get('problem_id'):<4} {row.get('arm', ''):<7} "
                f"turn {row.get('turn') or 0}  {row.get('category', ''):<11} "
                f"{row['observed_ms'] / 1000.0:.1f}s"
            )
    if under:
        lines.append("")
        lines.append(
            "no per-forward evidence (not benchmarked: only passing kernels are "
            "timed). Re-run to decide:"
        )
        by_category = Counter(row.get("category") for row in under)
        for category, count in sorted(by_category.items()):
            lines.append(f"  {category or '?':<11} {count}")
    if not audit["over"] and not under:
        lines.append("")
        lines.append("every record verified under budget; no re-run needed.")
    return lines


def format_chains(chains):
    lines = [f"{'p':>4} {'arm':<7} {'turn':>4}  {'category':<11} detail"]
    for (problem, arm), chain in chains.items():
        for attempt in chain:
            turn = attempt.get("turn") or 0
            detail = (attempt.get("detail") or "")[:45]
            lines.append(
                f"{problem:>4} {arm:<7} {turn:>4}  {attempt['category']:<11} {detail}"
            )
    return lines


def format_transition(chains):
    """Baseline (turn 0) against final, side by side."""
    baseline = distribution(chains, index=0)
    final = distribution(chains, index=-1)
    arms = sorted(set(baseline) | set(final))
    categories = category_columns({**baseline, **final})
    lines = [f"{'arm':<7} {'stage':<9}" + "".join(f"{name:>11}" for name in categories)]
    for arm in arms:
        for label, table in (("one-shot", baseline), ("repaired", final)):
            counts = table.get(arm, Counter())
            lines.append(
                f"{arm:<7} {label:<9}"
                + "".join(f"{counts.get(name, 0):>11}" for name in categories)
            )
    return lines


def format_false_pass(chains):
    lines = []
    for label, index in (("one-shot", 0), ("repaired", -1)):
        for arm, (silent, total, rate) in false_pass(
            distribution(chains, index=index)
        ).items():
            if rate is None:
                lines.append(f"{label:<9} {arm:<7} no standard-passing kernels")
            else:
                lines.append(
                    f"{label:<9} {arm:<7} {silent}/{total} = {rate:.0f}% silent"
                )
    return lines


def format_repair(chains):
    counts = repair_outcomes(chains)
    labels = {
        "no_repair_needed": "passed or silent on turn 0, no repair attempted",
        "no_repair_run": "failed on turn 0 but no repair recorded",
        "rescued": "repaired to pass",
        "hidden": "visible failure repaired into a SILENT failure",
        "still_failing": "still visibly failing after repair",
        "other": "other",
    }
    total_turns = sum(len(chain) for chain in chains.values())
    lines = [
        f"chains: {len(chains)}   attempts: {total_turns}   "
        f"repair attempts: {total_turns - len(chains)}"
    ]
    for key, text in labels.items():
        if counts.get(key):
            lines.append(f"  {counts[key]:>3}  {text}")
    return lines


def missing_chains(chains, problems, arms):
    """Problem-arms with no record at all; a timeout leaves nothing behind."""
    return sorted(
        (problem, arm)
        for problem in problems
        for arm in arms
        if (problem, arm) not in chains
    )


def format_warnings(rows, chains, *, problems=None, arms=None):
    """Surface data problems rather than hiding them."""
    lines = []
    duplicates = len(rows) - len(dedupe(rows))
    if duplicates:
        lines.append(
            f"{duplicates} duplicate (problem, arm, turn) records dropped "
            "-- older sessions or re-runs"
        )
    errored = [key for key, chain in chains.items() if chain[-1]["category"] == "error"]
    if errored:
        lines.append(f"{len(errored)} chains end in `error`: {errored}")
    if problems and arms:
        gaps = missing_chains(chains, problems, arms)
        if gaps:
            lines.append(
                f"{len(gaps)} of {len(problems) * len(arms)} problem-arms have NO "
                f"record (crash or timeout): {gaps}"
            )
    return lines
