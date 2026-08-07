"""Export the classification cache as flat, analysis-ready tables.

    uv run modal run export.py [--level 3] [--with-code]

Writes data/attempts.csv (one row per attempt), data/attempts.jsonl (the same
rows with nested fields intact), data/chains.csv (one row per repair chain),
and with --with-code the generated source per attempt.

The volume stores one hash-named JSON per attempt, which suits a cache and not
analysis. These tables are the archivable artifact and regenerate from the
cache at any time.
"""

import csv
import json
import os

import modal

from kfault.analysis import gpu_label, outcome
from kfault.config import RESULTS_DIR
from pipeline import image, results_volume

app = modal.App("kfault-export")

OUT = "data"

# Probes are a fixed set, so each gets its own column. Anything unexpected
# still lands in the JSONL, which keeps the nested battery_diagnostics.
PROBES = (
    "fresh_seed_1",
    "fresh_seed_2",
    "fresh_seed_3",
    "non_contig",
    "nondeterministic",
    "zeros",
    "small_magnitude",
    "extreme",
    "fp16",
)

COLUMNS = [
    # identity
    "model_id",
    "device",
    "gpu_config",
    "level",
    "problem_id",
    "problem_name",
    "arm",
    "reasoning_effort",
    "sample",
    "turn",
    # outcome
    "category",
    "failure_stage",
    "failure_type",
    "detail",
    "robustness_flags",
    # battery
    *[f"probe_{name}" for name in PROBES],
    "probes_oom_skipped",
    "sanitizer_status",
    # performance
    "ref_ms",
    "kernel_ms",
    "speedup",
    "forward_seconds",
    # provenance
    "prompt_tokens",
    "completion_tokens",
    "gen_seconds",
    "classify_seconds",
    "reused_response",
    "schema",
    "forward_time_limit",
    "seed",
    "labelled_at",
    "generation_key",
    "generation_artifact_key",
    "classification_key",
]


@app.function(image=image, volumes={RESULTS_DIR: results_volume}, timeout=1800)
def collect(level: int, with_code: bool):
    import glob

    results_volume.reload()
    rows = []
    for path in sorted(glob.glob(f"{RESULTS_DIR}/classifications/*.json")):
        with open(path) as file:
            record = json.load(file)
        if level and record.get("level") != level:
            continue
        config = record.get("classification_config") or {}
        failure = record.get("failure") or {}
        timing = record.get("timing") or {}
        usage = record.get("usage") or {}
        diagnostics = record.get("battery_diagnostics") or {}
        sanitizer = record.get("sanitizer") or {}

        row = {
            "model_id": record.get("model_id"),
            "device": gpu_label(record),
            "gpu_config": config.get("gpu"),
            "level": record.get("level"),
            "problem_id": record.get("problem_id"),
            "problem_name": record.get("problem_name"),
            "arm": record.get("arm"),
            "reasoning_effort": record.get("reasoning_effort"),
            "sample": record.get("sample"),
            "turn": record.get("turn") or 0,
            "category": record.get("category"),
            "failure_stage": failure.get("stage"),
            "failure_type": failure.get("error_type"),
            "detail": (record.get("detail") or "").replace("\n", " ")[:300],
            "robustness_flags": ",".join(record.get("robustness_flags") or []),
            "probes_oom_skipped": sum(
                1
                for v in diagnostics.values()
                if v.get("status") == "skipped_out_of_memory"
            ),
            "sanitizer_status": sanitizer.get("status"),
            "ref_ms": timing.get("ref_ms"),
            "kernel_ms": timing.get("kernel_ms"),
            "speedup": timing.get("speedup"),
            "forward_seconds": record.get("forward_seconds"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "gen_seconds": record.get("gen_seconds"),
            "classify_seconds": record.get("classify_seconds"),
            "reused_response": record.get("reused_response"),
            "schema": config.get("schema"),
            "forward_time_limit": config.get("forward_time_limit"),
            "seed": record.get("seed"),
            "labelled_at": record.get("labelled_at"),
            "generation_key": record.get("generation_key"),
            "generation_artifact_key": record.get("generation_artifact_key"),
            "classification_key": record.get("classification_key"),
        }
        for name in PROBES:
            row[f"probe_{name}"] = (diagnostics.get(name) or {}).get("status")
        if with_code:
            row["_code"] = record.get("generated_code") or ""
        row["_diagnostics"] = diagnostics
        rows.append(row)
    return rows


@app.local_entrypoint()
def main(level: int = 2, with_code: bool = False):
    rows = collect.remote(level, with_code)
    os.makedirs(OUT, exist_ok=True)

    # newest record per identity: re-runs leave older copies behind
    best = {}
    for row in rows:
        key = (
            row["model_id"],
            row["device"],
            row["forward_time_limit"],
            row["schema"],
            row["problem_id"],
            row["arm"],
            row["sample"],
            row["turn"],
        )
        if key not in best or (row["labelled_at"] or "") > (
            best[key]["labelled_at"] or ""
        ):
            best[key] = row
    rows = sorted(
        best.values(),
        key=lambda r: (
            r["model_id"] or "",
            r["device"] or "",
            r["problem_id"] or 0,
            r["arm"] or "",
            r["turn"],
        ),
    )

    with open(f"{OUT}/attempts.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with open(f"{OUT}/attempts.jsonl", "w") as file:
        for row in rows:
            payload = {k: v for k, v in row.items() if k != "_code"}
            file.write(json.dumps(payload) + "\n")

    chains = {}
    for row in rows:
        key = (row["model_id"], row["device"], row["problem_id"], row["arm"])
        chains.setdefault(key, []).append(row)
    for chain in chains.values():
        chain.sort(key=lambda r: r["turn"])

    chain_columns = [
        "model_id",
        "device",
        "problem_id",
        "problem_name",
        "arm",
        "turns",
        "first_category",
        "final_category",
        "outcome",
        "final_speedup",
    ]
    with open(f"{OUT}/chains.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=chain_columns)
        writer.writeheader()
        for (model, device, pid, arm), chain in sorted(chains.items()):
            writer.writerow(
                {
                    "model_id": model,
                    "device": device,
                    "problem_id": pid,
                    "problem_name": chain[0]["problem_name"],
                    "arm": arm,
                    "turns": len(chain),
                    "first_category": chain[0]["category"],
                    "final_category": chain[-1]["category"],
                    "outcome": outcome(chain),
                    "final_speedup": chain[-1]["speedup"],
                }
            )

    if with_code:
        os.makedirs(f"{OUT}/kernels", exist_ok=True)
        written = 0
        for row in rows:
            code = row.get("_code")
            if not code:
                continue
            slug = (row["model_id"] or "?").replace(".", "-").replace(":", "-")
            name = (
                f"{slug}_{row['device']}_p{row['problem_id']}"
                f"_{row['arm']}_t{row['turn']}.py"
            )
            with open(f"{OUT}/kernels/{name}", "w") as file:
                file.write(code)
            written += 1
        print(f"  kernels written : {written}")

    print(f"\nwrote {OUT}/attempts.csv   ({len(rows)} attempts)")
    print(f"wrote {OUT}/attempts.jsonl")
    print(f"wrote {OUT}/chains.csv     ({len(chains)} chains)")
    models = sorted({r["model_id"] for r in rows if r["model_id"]})
    devices = sorted({r["device"] for r in rows if r["device"]})
    print(f"\nmodels : {models}")
    print(f"devices: {devices}")
