"""Modal app: generate kernels, classify them, cache results.

Pure logic lives in the kfault package (config, prompts, generation,
sanitizer, battery, classifier, report). This file only wires it to Modal,
the LLM API and the cache volume.
"""

import modal
from modal.exception import FunctionTimeoutError

from kfault import generation as gen
from kfault.cache import content_hash, write_json_atomic
from kfault.classifier import classify
from kfault.config import (
    ARMS,
    CLASSIFIER_CACHE_VERSION,
    CLASSIFIER_TIMEOUT_S,
    CUDA_IMAGE,
    DATASET_REVISION,
    DEFAULT_FORWARD_TIME_LIMIT_S,
    FORWARD_TIME_LIMIT_S,
    GPU,
    MODEL_ID,
    PINNED_PACKAGES,
    RESULTS_DIR,
)
from kfault.prompts import build_prompt, strip_fences, strip_reasoning
from kfault.repair import build_feedback, build_repair_prompt, should_repair
from kfault.report import format_summary

app = modal.App("kfault-pipeline", include_source=False)

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .pip_install(*PINNED_PACKAGES)
    # Every KFAULT_* setting must be listed here. Modal does not forward local
    # environment variables, and these are read inside containers, so anything
    # omitted silently falls back to its default.
    .env(
        {
            "KFAULT_GPU": GPU,
            "KFAULT_MODEL": MODEL_ID,
            "KFAULT_FORWARD_LIMIT_S": str(FORWARD_TIME_LIMIT_S),
            "KFAULT_CLASSIFIER_TIMEOUT_S": str(CLASSIFIER_TIMEOUT_S),
        }
    )
    .add_local_python_source("kfault", "pipeline")
)

results_volume = modal.Volume.from_name("kfault-results", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    cpu=(2.0, 8.0),
    memory=(4096, 16384),
    # See CLASSIFIER_TIMEOUT_S. Not part of the cache key.
    timeout=CLASSIFIER_TIMEOUT_S,
    block_network=True,
    restrict_modal_access=True,
    single_use_containers=True,
)
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
    # Nothing torch-typed may cross the Modal boundary: the local caller has no
    # torch, so a propagating torch exception is replaced by "Deserialization
    # failed because the 'torch' module is not available" and the real cause is
    # lost. Convert any unhandled failure into a plain-data error result.
    try:
        category, detail, extra = classify(
            generated_code, problem_code, use_sanitizer=use_sanitizer, seed=seed
        )
    except BaseException as error:
        category, detail = "error", str(error)[:1000]
        extra = {
            "failure": {
                "stage": "classifier_uncaught",
                "error_type": type(error).__name__,
                "message": detail,
            }
        }
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    # Every value must be a PLAIN builtin. torch.__version__ is a TorchVersion
    # (str subclass): json.dump writes it happily, but Modal pickles the class,
    # and the local caller has no torch to unpickle it, so the whole result
    # fails to deserialize with "the 'torch' module is not available".
    extra["runtime_environment"] = {
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "gpu_name": str(properties.name),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "cuda_image": CUDA_IMAGE,
        "pinned_packages": list(PINNED_PACKAGES),
    }
    return category, detail, extra


def _load_problem(level: int, problem_num: int):
    from datasets import load_dataset

    ds = load_dataset(
        "ScalingIntelligence/KernelBench",
        split=f"level_{level}",
        revision=DATASET_REVISION,
    )
    return next(p for p in ds if p["problem_id"] == problem_num)


def _import_legacy(path_parts, key, config, prompt):
    """Find an old combined-cache entry matching this exact request."""
    import json
    from glob import glob

    level, problem_num, arm, sample, temperature = path_parts
    suffix = "" if sample == 0 else f"_s{sample}"
    pattern = f"{RESULTS_DIR}/level{level}_p{problem_num}_{arm}{suffix}*.json"
    for legacy_path in sorted(glob(pattern), reverse=True):
        with open(legacy_path) as file:
            legacy = json.load(file)
        if gen.legacy_matches(
            legacy, arm=arm, sample=sample, temperature=temperature, prompt=prompt
        ):
            return gen.from_legacy(legacy, legacy_path, key, config)
    return None


def _resolve_generation(
    *,
    prompt,
    key,
    config,
    path,
    path_parts,
    reasoning_effort,
    temperature,
    regen,
    allow_legacy=True,
):
    """Return ``(generation, reused)``, paying for a new response only if needed."""
    import json
    import os

    cached = None
    if os.path.exists(path):
        with open(path) as file:
            cached = json.load(file)

    if cached is None and not regen and allow_legacy:
        cached = _import_legacy(path_parts, key, config, prompt)
        if cached is not None:
            write_json_atomic(path, cached)
            results_volume.commit()

    if cached is not None and not regen:
        return cached, True

    generation = gen.request(
        prompt,
        api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        key=key,
        config=config,
    )
    write_json_atomic(path, generation)
    results_volume.commit()
    return generation, False


def _attempt(*, context, prompt, turn, feedback, force, regen, use_sanitizer, seed):
    """Generate (or reuse) one attempt and classify it."""
    import json
    import os
    import time
    from datetime import datetime, timezone

    from kfault.cache import cache_key

    level = context["level"]
    problem_num = context["problem_id"]
    arm = context["arm"]
    reasoning_effort = context["reasoning_effort"]
    sample = context["sample"]
    temperature = context["temperature"]
    code = context["problem_code"]
    name = context["problem_name"]

    generation_config = gen.generation_config(
        reasoning_effort=reasoning_effort,
        level=level,
        problem_id=problem_num,
        sample=sample,
        temperature=temperature,
        dataset_revision=DATASET_REVISION,
        prompt=prompt,
        turn=turn,
        feedback=feedback,
    )
    generation_key = gen.generation_key(generation_config)
    generation_path = f"{RESULTS_DIR}/generations/{generation_key}.json"

    # force alone re-classifies the cached response (no API spend);
    # regen forces a genuinely new generation
    generation, reused = _resolve_generation(
        prompt=prompt,
        key=generation_key,
        config=generation_config,
        path=generation_path,
        path_parts=(level, problem_num, arm, sample, temperature),
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        regen=regen,
        allow_legacy=turn == 0,
    )

    raw_response = generation.get("raw_response")
    finish_reason = generation.get("finish_reason")
    generation_artifact_key = generation.get("generation_artifact_key") or content_hash(
        raw_response or ""
    )

    classification_config = {
        "schema": CLASSIFIER_CACHE_VERSION,
        "generation_artifact_key": generation_artifact_key,
        "seed": seed,
        "use_sanitizer": use_sanitizer,
        "gpu": GPU,
        "cuda_image": CUDA_IMAGE,
        "packages": PINNED_PACKAGES,
    }
    # The budget decides runtime/forward_time_limit versus a full battery run,
    # so two budgets must not share an entry. Added only when non-default, so
    # the default keeps its existing hash and cached entries stay valid.
    if FORWARD_TIME_LIMIT_S != DEFAULT_FORWARD_TIME_LIMIT_S:
        classification_config["forward_time_limit"] = FORWARD_TIME_LIMIT_S
    classification_key = cache_key(classification_config)
    classification_path = f"{RESULTS_DIR}/classifications/{classification_key}.json"
    if os.path.exists(classification_path) and not force and not regen:
        with open(classification_path) as file:
            cached_classification = json.load(file)
        cached_classification["cached"] = True
        return cached_classification

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
        try:
            category, detail, extra = run_classifier.spawn(
                generated_code, code, use_sanitizer, seed
            ).get()
        except FunctionTimeoutError as error:
            # The kernel hung inside a call we cannot interrupt. Recorded as a
            # real result rather than allowed to escape: an escaping exception
            # leaves no row, skips the cache write, and kills the repair chain.
            category = "runtime"
            detail = (
                f"classification exceeded the {CLASSIFIER_TIMEOUT_S}s container "
                f"limit; the kernel is hung or pathologically slow"
            )
            extra = {
                "failure": {
                    "stage": "container_timeout",
                    "error_type": type(error).__name__,
                    "message": str(error)[:1000],
                }
            }
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
        "dataset_revision": DATASET_REVISION,
        "generation_key": generation_key,
        "generation_artifact_key": generation_artifact_key,
        "generation_config": generation_config,
        "classification_key": classification_key,
        "classification_config": classification_config,
        "category": category,
        "detail": detail,
        "finish_reason": finish_reason,
        "prompt": prompt,
        "raw_response": raw_response,
        "generated_code": generated_code,
        "usage": generation.get("usage"),
        "resolved_model": generation.get("resolved_model"),
        "gen_seconds": generation.get("gen_seconds"),
        "generated_at": generation.get("generated_at"),
        "classify_seconds": classify_seconds,
        "labelled_at": datetime.now(timezone.utc).isoformat(),
        "reused_response": bool(reused),
        "cached": False,
        **extra,
    }
    # Only present on repair runs, so one-shot artifacts stay byte-identical.
    if turn:
        result["turn"] = turn
        result["repair_feedback"] = feedback

    write_json_atomic(classification_path, result)
    results_volume.commit()
    return result


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("bedrock")],
    volumes={RESULTS_DIR: results_volume},
    # generous: a repair run pays generation + classification once per turn
    timeout=7200,
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
    repair_turns: int = 0,
):
    """CPU orchestrator: cache, generation, then GPU classification.

    Runs on CPU so no GPU sits idle while the LLM thinks; the GPU container
    (run_classifier) starts only when the kernel code is ready.

    repair_turns=0 is the one-shot KernelBench protocol. Above 0, a failed
    attempt is fed back to the model (visible stages only) and retried.
    """
    import os

    results_volume.reload()
    os.makedirs(f"{RESULTS_DIR}/generations", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/classifications", exist_ok=True)

    problem = _load_problem(level, problem_num)
    context = {
        "level": level,
        "problem_id": problem_num,
        "problem_name": problem["name"],
        "problem_code": problem["code"],
        "arm": arm,
        "reasoning_effort": ARMS[arm],
        "sample": sample,
        "temperature": temperature,
    }

    prompt = build_prompt(problem["code"])
    feedback = ""
    history = []
    result = None

    for turn in range(repair_turns + 1):
        if turn:
            prompt = build_repair_prompt(
                problem["code"], result.get("generated_code"), feedback
            )
        result = _attempt(
            context=context,
            prompt=prompt,
            turn=turn,
            feedback=feedback,
            force=force,
            regen=regen,
            use_sanitizer=use_sanitizer,
            seed=seed,
        )
        history.append(
            {
                "turn": turn,
                "category": result["category"],
                "detail": result.get("detail"),
                "generation_key": result["generation_key"],
                "classification_key": result["classification_key"],
            }
        )
        if not should_repair(result["category"]):
            break
        feedback = build_feedback(result["category"], result.get("detail"), result)
        if not feedback:
            break

    if repair_turns:
        result = {
            **result,
            "repair_turns": repair_turns,
            "turns_used": len(history) - 1,
            "repair_history": history,
        }
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
    seed: int = 42,
    repair_turns: int = 0,
):
    """Fan out problems x arms x samples across Modal containers.

    Arms: think (reasoning high) vs direct (reasoning none), same model.
    Sample 0 is greedy (temp 0); samples 1..n-1 use --temperature.
    --force re-classifies cached generations for free; add --regen for
    genuinely new generations.
    --repair-turns N lets each attempt see its own error and retry N times.
    """
    import itertools
    from collections import Counter

    nums = _parse_problems(problems)

    handles = {}
    for num, arm, s in itertools.product(nums, ARMS, range(samples)):
        temp = 0.0 if s == 0 else temperature
        handles[(num, arm, s)] = diagnose.spawn(
            num, arm, level, force, use_sanitizer, s, temp, regen, seed, repair_turns
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
    print(format_summary(level, nums, samples, ARMS, rows, per_arm))
