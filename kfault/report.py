"""Run summary formatting.

Pure text assembly, no Modal and no I/O, so the report layout can be tested
without spending a run.
"""

CANONICAL_CATEGORIES = ["syntactic", "runtime", "functional", "silent", "pass"]


def category_columns(per_arm: dict) -> list:
    """Canonical categories first, then any extra category actually seen."""
    seen = set()
    for counts in per_arm.values():
        seen |= set(counts)
    return CANONICAL_CATEGORIES + sorted(seen - set(CANONICAL_CATEGORIES))


def format_rows(level: int, rows) -> list:
    """One line per (problem, arm, sample) result."""
    lines = []
    for num, arm, sample, result in rows:
        cache = " (cached)" if result.get("cached") else ""
        sanitizer_status = (result.get("sanitizer") or {}).get("status")
        sanitizer = f" san={sanitizer_status}" if sanitizer_status else ""
        detail = (result.get("detail") or "")[:60]
        lines.append(
            f"L{level} p{num:<3} [{arm} s{sample}] {result['category']:<10} "
            f"{detail}{sanitizer}{cache}"
        )
    return lines


def format_distribution(arms, per_arm: dict) -> list:
    """Category counts per arm, one column per category."""
    categories = category_columns(per_arm)
    lines = ["arm     " + "".join(f"{name:>11}" for name in categories)]
    for arm in arms:
        counts = per_arm[arm]
        lines.append(
            f"{arm:<7} " + "".join(f"{counts.get(name, 0):>11}" for name in categories)
        )
    return lines


def format_false_pass(arms, per_arm: dict) -> list:
    """RQ2 headline: silent share of standard-passing kernels."""
    lines = []
    for arm in arms:
        silent = per_arm[arm].get("silent", 0)
        passed = per_arm[arm].get("pass", 0)
        denominator = silent + passed
        if denominator:
            percent = 100 * silent / denominator
            lines.append(
                f"{arm}: {silent}/{denominator} = {percent:.0f}% of standard-passing "
                "kernels are silent failures"
            )
        else:
            lines.append(f"{arm}: no standard-passing kernels")
    return lines


def format_summary(
    level: int, problems, samples: int, arms, rows, per_arm: dict
) -> str:
    """Assemble the full end-of-run report."""
    lines = [
        "",
        f"=== Level {level} | problems {problems} | {samples} sample(s) ===",
        *format_rows(level, rows),
        "",
        "--- category distribution (by arm) ---",
        *format_distribution(arms, per_arm),
        "",
        "--- false-pass rate (RQ2) ---",
        *format_false_pass(arms, per_arm),
    ]
    return "\n".join(lines)
