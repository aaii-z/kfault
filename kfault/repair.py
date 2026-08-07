"""Feedback for multi-turn kernel repair.

Only stages 1-3 are reported back to the model. Stage 4 (silent) stays
hidden: leaking the stress battery would let the model fit the held-out test
and destroy the false-pass measurement (RQ2).
"""

from kfault.config import CLASSIFIER_TIMEOUT_S
from kfault.prompts import build_prompt

# Categories the model is allowed to see and retry. `silent` and `pass` end
# the loop; `error` is a harness fault, not the kernel's.
REPAIRABLE = {"syntactic", "runtime", "functional", "empty"}


def should_repair(category: str) -> bool:
    return category in REPAIRABLE


def tail(text, limit: int = 800) -> str:
    """Keep the END of a log: nvcc puts the decisive line last."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return "...\n" + text[-limit:]


def build_feedback(category: str, detail, extra: dict) -> str:
    """One compact report of what went wrong, or "" if nothing to say."""
    if not should_repair(category):
        return ""

    failure = (extra or {}).get("failure") or {}
    stage = failure.get("stage")
    message = failure.get("message") or detail

    if category == "empty":
        return "Your previous answer contained no code block."

    # A timeout's stored message is the harness's own and says nothing the
    # model can act on. Describe what happened instead.
    if stage == "container_timeout":
        return (
            "Your previous kernel did not finish and was killed.\n"
            f"A single classification is allowed {CLASSIFIER_TIMEOUT_S}s, which "
            "covers compilation plus roughly 90 forward calls. Your kernel did "
            "not get through that.\n"
            "This means it is either hung (an infinite or unbounded loop, a "
            "barrier some threads never reach, or a wait that never ends) or "
            "so slow it cannot be measured.\n"
            "Rewrite it so one forward call finishes fast. Check every loop "
            "bound and exit condition, make sure all threads reach any "
            "__syncthreads(), and do not serialize work onto one thread."
        )

    if stage == "forward_time_limit":
        return (
            "Your previous kernel produced no result because it is far too "
            "slow.\n"
            f"{message}\n"
            "It was correct enough to run, so the logic may be fine; the "
            "problem is speed. Use enough parallelism, coalesce global memory "
            "accesses, avoid per-thread loops over large dimensions, and do "
            "not let a single thread do the whole computation."
        )

    lines = [f"Your previous kernel failed at stage: {category}"]
    if stage:
        lines.append(f"Failing step: {stage}")

    if category == "syntactic":
        lines.append("Build error:")
    elif category == "runtime":
        lines.append("Runtime error:")
    else:
        lines.append("Output did not match the reference:")
    lines.append(tail(message))

    sanitizer = (extra or {}).get("sanitizer") or {}
    if category == "runtime" and sanitizer.get("log_excerpt"):
        lines.append("compute-sanitizer:")
        lines.append(tail(sanitizer["log_excerpt"], 400))

    return "\n".join(lines)


def build_repair_prompt(problem_code: str, previous_code: str, feedback: str) -> str:
    """Restate the task, show the failed attempt, then the error."""
    return (
        build_prompt(problem_code)
        + "\n\nYou already tried this, and it failed:\n\n```python\n"
        + (previous_code or "")
        + "\n```\n\n"
        + feedback
        + "\n\nFix the problem and output the complete corrected ModelNew code "
        "in one code block. No explanation, no testing code."
    )
