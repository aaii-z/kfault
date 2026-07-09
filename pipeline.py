"""Modal app: generate kernels, classify them, cache results.

Pure logic lives in the kfault package (prompts, sanitizer, battery,
classifier). This file only wires it to Modal, the LLM API and the cache.
"""

import os

import modal

from kfault.classifier import classify
from kfault.prompts import build_prompt, strip_fences, strip_reasoning

app = modal.App("kfault-pipeline")

# Provider: AWS Bedrock, OpenAI-compatible endpoint, DeepSeek V3.2 serverless.
# Two arms, SAME model weights, only the reasoning switch differs. This is the
# controlled contrast the study is built on.
BEDROCK_URL = "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"
MODEL_ID = "deepseek.v3-v1:0"
ARMS = {
    "think": "high",   # reasoning ON
    "direct": "none",  # reasoning OFF
}

# GPU for all containers. Override per run, e.g. for the RQ4 multi-GPU pass
# or for L4/A100 where compute-sanitizer works:
#   KFAULT_GPU=L4 uv run modal run pipeline.py ...
GPU = os.environ.get("KFAULT_GPU", "T4")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("torch", "ninja", "openai>=1.0.0", "datasets>=2.0.0")
    .add_local_python_source("kfault")
)

# one JSON per (level, problem, model, sample) so reruns never re-spend
results_volume = modal.Volume.from_name("kfault-results", create_if_missing=True)
RESULTS_DIR = "/results"


@app.function(image=image, gpu=GPU, timeout=900)
def run_classifier(
    generated_code: str,
    problem_code: str,
    use_sanitizer: bool = True,
    seed: int = 42,
):
    """GPU stage only: classify an already-generated kernel."""
    import torch

    # full fp32, no TF32: Ampere+ GPUs otherwise use TF32 in the reference
    # and produce spurious functional failures
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return classify(
        generated_code, problem_code, use_sanitizer=use_sanitizer, seed=seed
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("bedrock")],
    volumes={RESULTS_DIR: results_volume},
    timeout=1800,
)
def diagnose(
    problem_num: int = 1,
    arm: str = "direct",
    level: int = 1,
    force: bool = False,
    use_sanitizer: bool = True,
    sample: int = 0,
    temperature: float = 0.0,
    regen: bool = False,
    seed: int = 42,
):
    """CPU orchestrator: cache, generation, then GPU classification.

    Runs on CPU so no GPU sits idle while the LLM thinks; the GPU container
    (run_classifier) starts only when the kernel code is ready.
    """
    import json
    import os
    import time
    from datetime import datetime, timezone

    from datasets import load_dataset
    from openai import OpenAI

    reasoning_effort = ARMS[arm]
    suffix = "" if sample == 0 else f"_s{sample}"
    path = f"{RESULTS_DIR}/level{level}_p{problem_num}_{arm}{suffix}.json"

    results_volume.reload()
    cached = None
    if os.path.exists(path):
        with open(path) as f:
            cached = json.load(f)
    if cached is not None and not force:
        cached["cached"] = True
        return cached

    ds = load_dataset("ScalingIntelligence/KernelBench", split=f"level_{level}")
    problem = next(p for p in ds if p["problem_id"] == problem_num)
    code, name = problem["code"], problem["name"]

    prompt = build_prompt(code)

    # force alone re-classifies the cached response (no API spend);
    # regen forces a genuinely new generation
    reused = cached is not None and not regen and cached.get("raw_response")
    if reused:
        raw_response = cached["raw_response"]
        finish_reason = cached.get("finish_reason")
        temperature = cached.get("temperature", 0.0)
        # provenance of the ORIGINAL generation survives re-labels
        usage = cached.get("usage")
        gen_seconds = cached.get("gen_seconds")
        generated_at = cached.get("generated_at")
    else:
        client = OpenAI(
            api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
            base_url=BEDROCK_URL,
        )
        t0 = time.monotonic()
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            # generous cap: the think arm spends tokens before the code
            max_tokens=16000,
            extra_body={"reasoning_effort": reasoning_effort},
        )
        gen_seconds = round(time.monotonic() - t0, 2)
        generated_at = datetime.now(timezone.utc).isoformat()
        raw_response = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        usage = response.usage.model_dump() if response.usage else None

    # no content, or thinking consumed the whole budget: generation failure,
    # distinct `empty` category
    answer = strip_reasoning(raw_response) if raw_response else ""
    generated_code = strip_fences(answer) if answer else None
    if not answer:
        category, detail, extra = "empty", f"no content (finish={finish_reason})", {}
        classify_seconds = None
    else:
        t0 = time.monotonic()
        # spawn + get: detached call, survives client disconnects (Modal
        # warns that plain .remote() may be canceled mid-run)
        category, detail, extra = run_classifier.spawn(
            generated_code, code, use_sanitizer, seed
        ).get()
        # includes GPU container start-up, not pure classification time
        classify_seconds = round(time.monotonic() - t0, 2)

    result = {
        "problem_name": name,
        "level": level,
        "problem_id": problem_num,
        "model_id": MODEL_ID,
        "arm": arm,
        "reasoning_effort": reasoning_effort,
        "sample": sample,
        "temperature": temperature,
        "seed": seed,
        "category": category,
        "detail": detail,
        "finish_reason": finish_reason,
        "prompt": prompt,
        "raw_response": raw_response,
        "generated_code": generated_code,
        "usage": usage,
        "gen_seconds": gen_seconds,
        "generated_at": generated_at,
        "classify_seconds": classify_seconds,
        "labelled_at": datetime.now(timezone.utc).isoformat(),
        "reused_response": bool(reused),
        "cached": False,
        **extra,
    }

    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()
    return result


def _parse_problems(spec: str):
    """Expand "1,2,5-8" into a sorted list of ints."""
    nums: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            nums.update(range(int(lo), int(hi) + 1))
        else:
            nums.add(int(part))
    return sorted(nums)


@app.local_entrypoint()
def main(
    level: int = 2,
    problems: str = "1-10",
    samples: int = 1,
    temperature: float = 0.7,
    force: bool = False,
    regen: bool = False,
    use_sanitizer: bool = True,
):
    """Fan out problems x arms x samples across Modal containers.

    Arms: think (reasoning high) vs direct (reasoning none), same model.
    Sample 0 is greedy (temp 0); samples 1..n-1 use --temperature.
    --force re-classifies cached generations for free; add --regen for
    genuinely new generations.
    """
    import itertools
    from collections import Counter

    nums = _parse_problems(problems)

    handles = {}
    for num, arm, s in itertools.product(nums, ARMS, range(samples)):
        temp = 0.0 if s == 0 else temperature
        handles[(num, arm, s)] = diagnose.spawn(
            num, arm, level, force, use_sanitizer, s, temp, regen
        )

    rows = []
    per_arm = {arm: Counter() for arm in ARMS}
    for (num, arm, s), h in handles.items():
        # one failed container must not sink the whole aggregation
        try:
            r = h.get()
        except Exception as e:
            r = {"category": "error", "detail": str(e)[:80]}
        rows.append((num, arm, s, r))
        per_arm[arm][r["category"]] += 1

    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    print(f"\n=== Level {level} | problems {nums} | {samples} sample(s) ===")
    for num, arm, s, r in rows:
        cache = " (cached)" if r.get("cached") else ""
        san = r.get("sanitizer_errors")
        san_s = f" san={san}" if san else ""
        detail = (r.get("detail") or "")[:60]
        print(
            f"L{level} p{num:<3} [{arm} s{s}] {r['category']:<10} "
            f"{detail}{san_s}{cache}"
        )

    print("\n--- category distribution (by arm) ---")
    canonical = ["syntactic", "runtime", "functional", "silent", "pass"]
    seen = set().union(*[set(c) for c in per_arm.values()])
    cats = canonical + sorted(seen - set(canonical))
    header = "arm     " + "".join(f"{c:>11}" for c in cats)
    print(header)
    for arm in ARMS:
        line = f"{arm:<7} " + "".join(f"{per_arm[arm].get(c, 0):>11}" for c in cats)
        print(line)

    # RQ2 headline: silent share of standard-passing kernels
    print("\n--- false-pass rate (RQ2) ---")
    for arm in ARMS:
        silent = per_arm[arm].get("silent", 0)
        passed = per_arm[arm].get("pass", 0)
        denom = silent + passed
        if denom:
            pct = 100 * silent / denom
            print(
                f"{arm}: {silent}/{denom} = {pct:.0f}% of standard-passing "
                "kernels are silent failures"
            )
        else:
            print(f"{arm}: no standard-passing kernels")
