"""Read the classification cache off the Modal volume and print the tables.

    uv run modal run analyze.py [--batch 1] [--problems 1,11,21]
                               [--gpu T4] [--chains] [--budget]

Read-only, and runs on CPU.
"""

from collections import Counter

import modal

from kfault.analysis import (
    build_chains,
    dedupe,
    format_chains,
    format_false_pass,
    format_forward_budget,
    format_repair,
    format_transition,
    format_warnings,
    forward_budget_audit,
    gpu_label,
    select,
)
from kfault.config import (
    ARMS,
    CLASSIFIER_CACHE_VERSION,
    FORWARD_TIME_LIMIT_S,
    MODEL_ID,
    RESULTS_DIR,
)
from pipeline import _parse_problems, image, results_volume

app = modal.App("kfault-analyze")

BATCH_DIR = "experiments/level2-repair/batches"


@app.function(image=image, volumes={RESULTS_DIR: results_volume}, timeout=600)
def collect():
    """Return every classification record as plain data."""
    import glob
    import json

    results_volume.reload()
    rows = []
    for path in sorted(glob.glob(f"{RESULTS_DIR}/classifications/*.json")):
        with open(path) as file:
            record = json.load(file)
        # keep the report small: drop the code and prompt blobs
        for key in ("prompt", "raw_response", "generated_code", "repair_feedback"):
            record.pop(key, None)
        rows.append(record)
    return rows


def _batch_problems(batch: int):
    path = f"{BATCH_DIR}/batch{batch:02d}.txt"
    with open(path) as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                return _parse_problems(line)
    raise SystemExit(f"no problem spec found in {path}")


@app.local_entrypoint()
def main(
    level: int = 2,
    problems: str = "",
    batch: int = 0,
    model_id: str = MODEL_ID,
    schema: int = CLASSIFIER_CACHE_VERSION,
    gpu: str = "",
    chains: bool = False,
    budget: bool = False,
):
    """Print baseline-vs-repaired tables for the selected records."""
    wanted = None
    if batch:
        wanted = _batch_problems(batch)
    elif problems:
        wanted = _parse_problems(problems)

    rows = select(
        collect.remote(),
        level=level,
        problems=wanted,
        model_id=model_id,
        schema=schema,
        gpu=gpu or None,
    )
    if not rows:
        raise SystemExit(
            f"no records match level={level} model={model_id} schema={schema} "
            f"gpu={gpu or 'any'} problems={wanted or 'all'}"
        )

    grouped = build_chains(dedupe(rows))
    scope = f"batch {batch:02d}" if batch else (problems or "all problems")

    print(f"\n=== Level {level} | {scope} | {model_id} | schema {schema} ===")
    if gpu:
        print(f"gpu filter: {gpu}")
    # Always show the real device mix: a table silently mixing two GPUs is not
    # comparable, and the config field is unreliable on older records.
    devices = Counter(gpu_label(row) or "unknown" for row in dedupe(rows))
    print("device: " + ", ".join(f"{name} x{n}" for name, n in devices.most_common()))
    if len(devices) > 1:
        print("WARNING: records span more than one device; results are not comparable")

    warnings = format_warnings(rows, grouped, problems=wanted, arms=list(ARMS))
    if warnings:
        print("\n--- warnings ---")
        for line in warnings:
            print(line)

    if chains:
        print("\n--- attempts ---")
        for line in format_chains(grouped):
            print(line)

    print("\n--- one-shot vs repaired (by arm) ---")
    for line in format_transition(grouped):
        print(line)

    print("\n--- false-pass rate (RQ2) ---")
    for line in format_false_pass(grouped):
        print(line)

    print("\n--- repair outcomes ---")
    for line in format_repair(grouped):
        print(line)

    if budget:
        # Which cached records predate the per-forward limit and would change?
        print("\n--- forward-budget audit ---")
        audit = forward_budget_audit(dedupe(rows), FORWARD_TIME_LIMIT_S)
        for line in format_forward_budget(audit, FORWARD_TIME_LIMIT_S):
            print(line)
