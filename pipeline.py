"""kfault diagnostic pipeline: classify an LLM-generated CUDA kernel into a
failure category — syntactic / runtime / functional / silent / pass.

This is the core of the dissertation. `demo.py` stays as the minimal reference;
this file is the evolving pipeline.

Categories (first matching stage wins):
  syntactic  - generated code fails to compile / build (nvcc or Python error)
  runtime    - compiles, but crashes OR trips compute-sanitizer on standard input
  functional - runs, but wrong output on the standard input (basic allclose)
  silent     - passes the standard check, but WRONG under the stress battery
  pass       - correct on standard input AND survives the whole battery

The classifier is split from the generator so it can be exercised directly by
the T5 validation harness (`validate.py`) with hand-crafted kernels of known
category, without spending any API budget.
"""

import modal

app = modal.App("kfault-pipeline")

# The CUDA *devel* image ships compute-sanitizer under /usr/local/cuda/bin, so
# Stage 2b needs no extra install — just torch/ninja to build the kernels.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("torch", "ninja", "openai>=1.0.0", "datasets>=2.0.0")
)

# persistent store for per-run results: one JSON per (level, problem, model) so
# reruns don't re-spend API/GPU (the "cache results" risk in the plan).
results_volume = modal.Volume.from_name("kfault-results", create_if_missing=True)
RESULTS_DIR = "/results"

# ---- Official KernelBench prompt --------------------------------------------
# This is the published KernelBench baseline prompt (backend=cuda, option=
# one_shot, precision=fp32), assembled exactly as
# src/kernelbench/prompt_constructor_toml.py does from prompts.toml:
#   problem_statement -> one-shot example -> arch -> precision note -> instruction
# The one-shot example is the canonical element-wise-add pair
# (prompts/model_ex_add.py + prompts/model_new_ex_add.py).
#
# Using the official prompt keeps our generations comparable to published
# KernelBench pass rates (which our silent-failure thesis argues against).
# Built by concatenation rather than str.format so the example's CUDA braces
# `{ ... }` are never parsed as format fields.

_EXAMPLE_ARCH = '''import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []'''

_EXAMPLE_NEW_ARCH = '''import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_add_cpp_source = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)'''

_PROBLEM_STATEMENT = (
    "You write custom CUDA operators to replace the pytorch operators in the "
    "given architecture to get speedups.\n\n"
    "You have complete freedom to choose the set of operators you want to "
    "replace. You may make the decision to replace some operators with custom "
    "CUDA operators and leave others unchanged. You may replace multiple "
    "operators with custom implementations, consider operator fusion "
    "opportunities (combining multiple operators into a single kernel, for "
    "example, combining matmul+relu), or algorithmic changes (such as online "
    "softmax). You are only limited by your imagination."
)

_INSTRUCTION = (
    "Optimize the architecture named Model with custom CUDA operators! Name "
    "your optimized output architecture ModelNew. Output the new code in "
    "codeblocks. Please generate real code, NOT pseudocode, make sure the code "
    "compiles and is fully functional. Just output the new model code, no "
    "other text, and NO testing code!"
)


def build_prompt(ref_arch_src: str) -> str:
    """Assemble the official KernelBench one-shot CUDA/fp32 prompt."""
    return (
        _PROBLEM_STATEMENT
        + "\n\nHere's an example to show you the syntax of inline embedding "
        "custom CUDA operators in PyTorch:\n\n"
        "Example:\n\n"
        "Input architecture:\n\n" + _EXAMPLE_ARCH + "\n\n"
        "Optimized with CUDA operators:\n\n" + _EXAMPLE_NEW_ARCH + "\n\n"
        "You are given the following architecture:\n\n\n" + ref_arch_src + "\n\n"
        "Note: The kernels should be optimized for FP32 (32-bit floating "
        "point) precision.\n\n" + _INSTRUCTION
    )


def _strip_fences(text: str) -> str:
    """Extract the first fenced code block, mirroring KernelBench's
    extract_first_code: drop a leading language tag (```python) and ignore any
    prose the model adds after the closing fence."""
    if "```" not in text:
        return text.strip()
    block = text.split("```")[1]
    # strip a leading language hint line (e.g. "python") if present
    first_line, _, rest = block.partition("\n")
    if first_line.strip().isalpha():
        block = rest
    return block.strip()


# ---- Stage 2b: compute-sanitizer (memcheck) ---------------------------------
# Driver executed *under* compute-sanitizer in a subprocess. Because the torch
# cpp_extension build is content-hashed and cached in TORCH_EXTENSIONS_DIR, the
# kernel that Stage 1 already compiled in-process is reused here (no recompile),
# so this is a cheap extra forward pass under memory checking.
_SANITIZER_DRIVER = r'''
import os
import torch

problem = open(os.environ["KFAULT_PROBLEM"]).read()
kernel = open(os.environ["KFAULT_KERNEL"]).read()

pns = {}
exec(problem, pns)
Model = pns["Model"]
get_init_inputs = pns.get("get_init_inputs", lambda: [])
get_inputs = pns.get("get_inputs", lambda: [])
init_inputs = get_init_inputs()

kns = {}
exec(kernel, kns)
ModelNew = kns["ModelNew"]
kernel_model = ModelNew(*init_inputs).cuda().eval()

inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in get_inputs()]
with torch.no_grad():
    kernel_model(*inputs)
torch.cuda.synchronize()
'''


def _run_sanitizer(problem_code: str, kernel_code: str):
    """Run the kernel once under compute-sanitizer memcheck.

    Returns (errors: int | None, detail: str). `errors is None` means the
    sanitizer could not be run (binary missing / launch failure); in that case
    the caller degrades gracefully and does not label a runtime failure.
    """
    import os
    import re
    import shutil
    import subprocess
    import tempfile

    sanitizer = shutil.which("compute-sanitizer") or (
        "/usr/local/cuda/bin/compute-sanitizer"
        if os.path.exists("/usr/local/cuda/bin/compute-sanitizer")
        else None
    )
    if sanitizer is None:
        return None, "compute-sanitizer not found"

    with tempfile.TemporaryDirectory() as tmp:
        prob_path = os.path.join(tmp, "problem.py")
        kern_path = os.path.join(tmp, "kernel.py")
        drv_path = os.path.join(tmp, "driver.py")
        with open(prob_path, "w") as f:
            f.write(problem_code)
        with open(kern_path, "w") as f:
            f.write(kernel_code)
        with open(drv_path, "w") as f:
            f.write(_SANITIZER_DRIVER)

        env = dict(os.environ, KFAULT_PROBLEM=prob_path, KFAULT_KERNEL=kern_path)
        try:
            proc = subprocess.run(
                [
                    sanitizer,
                    "--tool", "memcheck",
                    "--error-exitcode", "1",
                    "--target-processes", "all",
                    "python", drv_path,
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except Exception as e:  # launch failure / timeout — degrade gracefully
            return None, f"sanitizer launch failed: {str(e)[:200]}"

        out = (proc.stdout or "") + (proc.stderr or "")

        # compute-sanitizer cannot run in every environment — notably Modal's
        # virtualized T4 rejects it with "Device not supported". These are
        # TOOL failures, not kernel memory errors: degrade gracefully (return
        # None) so the kernel is still classified by the other stages instead
        # of being mislabelled `runtime`.
        unavailable = (
            "Device not supported",
            "not supported by",
            "Failed to initialize",
            "does not have permission",
            "Unsupported",
        )
        for marker in unavailable:
            if marker in out:
                return None, f"sanitizer unavailable ({marker})"

        m = re.search(r"ERROR SUMMARY:\s+(\d+)\s+error", out)
        if m:
            errors = int(m.group(1))
        else:
            # No summary line: infer from exit code. Non-zero here only when
            # --error-exitcode fired, i.e. sanitizer saw an error.
            errors = 1 if proc.returncode != 0 else 0

        # Only trust a positive count as a real memory violation if an actual
        # violation block is present; otherwise it's another tool-level hiccup.
        violation = any(
            k in out
            for k in ("Invalid ", "misaligned", "out-of-bounds", "Fatal", "leaked")
        )
        if errors and not violation:
            return None, "sanitizer reported non-memory error; ignoring"

        snippet = ""
        idx = out.find("Invalid ")
        if idx == -1:
            idx = out.find("========= ")
        if idx != -1:
            snippet = out[idx:idx + 800].replace("\n", " ")
        return errors, snippet


# ---- The classifier (shared by `diagnose` and the T5 validation harness) ----
def classify(
    generated_code: str,
    problem_code: str,
    *,
    use_sanitizer: bool = True,
):
    """Run Stages 1–4 on an already-obtained kernel and return
    (category, detail, extra) where extra holds diagnostics (robustness flags,
    sanitizer status). No API calls, no persistence — pure classification, so
    the validation harness can feed it known-category kernels.
    """
    import torch

    extra: dict = {}

    # ---- Stage 1: SYNTACTIC — does the generated code build? ---------------
    kernel_ns: dict = {}
    problem_ns: dict = {}
    try:
        exec(problem_code, problem_ns)
        Model = problem_ns["Model"]
        get_init_inputs = problem_ns.get("get_init_inputs", lambda: [])
        get_inputs = problem_ns.get("get_inputs", lambda: [])
        init_inputs = get_init_inputs()
    except Exception as e:
        # a broken *problem* is our bug, not the model's — surface loudly
        raise RuntimeError(f"reference problem failed to load: {e}") from e

    def fresh_inputs():
        return [
            x.cuda() if isinstance(x, torch.Tensor) else x for x in get_inputs()
        ]

    ref_model = Model(*init_inputs).cuda().eval()

    try:
        exec(generated_code, kernel_ns)          # load_inline compiles here
        ModelNew = kernel_ns["ModelNew"]
        kernel_model = ModelNew(*init_inputs).cuda().eval()
    except Exception as e:
        return "syntactic", str(e)[:500], extra

    # ---- Stage 2a: RUNTIME — does it crash on standard input? --------------
    std_inputs = fresh_inputs()
    try:
        with torch.no_grad():
            kernel_out = kernel_model(*std_inputs)
            ref_out = ref_model(*std_inputs)
        torch.cuda.synchronize()  # surface async kernel faults (illegal mem) here
    except Exception as e:
        return "runtime", str(e)[:500], extra

    # ---- Stage 2b: RUNTIME — memory errors under compute-sanitizer? --------
    # Catches illegal / out-of-bounds accesses that happen to *not* crash the
    # in-process run but are still undefined behaviour (the plan's "races,
    # illegal mem" runtime bucket).
    if use_sanitizer:
        errors, detail = _run_sanitizer(problem_code, generated_code)
        extra["sanitizer_errors"] = errors
        if errors:
            return "runtime", f"compute-sanitizer: {errors} error(s) {detail}", extra

    # ---- Stage 3: FUNCTIONAL — correct on the standard input? --------------
    # a wrong-shaped output makes allclose raise; that is still a functional
    # failure (correct execution, wrong result), not a runtime crash.
    if ref_out.shape != kernel_out.shape:
        return (
            "functional",
            f"shape mismatch {tuple(kernel_out.shape)} vs {tuple(ref_out.shape)}",
            extra,
        )
    if not torch.allclose(ref_out.float(), kernel_out.float(), atol=1e-3, rtol=1e-3):
        return "functional", "mismatch on standard input", extra

    # ---- Stage 4: SILENT — wrong under the fixed stress battery? -----------
    core, robust = _silent_battery(
        ref_model, kernel_model, fresh_inputs, init_inputs, Model
    )
    extra["robustness_flags"] = robust
    if core:
        return "silent", ", ".join(core), extra

    detail = "survived core battery" + (
        f" (robustness flags: {', '.join(robust)})" if robust else ""
    )
    return "pass", detail, extra


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("openrouter")],
    volumes={RESULTS_DIR: results_volume},
    timeout=1200,
)
def diagnose(
    problem_num: int = 1,
    model_id: str = "deepseek/deepseek-chat",
    level: int = 1,
    force: bool = False,
    use_sanitizer: bool = True,
):
    import json
    import os

    import torch
    from datasets import load_dataset
    from openai import OpenAI

    # fair correctness comparison: force full fp32 (no TF32) so the PyTorch
    # reference uses the same precision class as a naive kernel. Without this,
    # Ampere+ GPUs (L4/A100, used in the RQ4 multi-GPU pass) silently use TF32
    # and produce spurious functional failures.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    path = (
        f"{RESULTS_DIR}/level{level}_p{problem_num}_"
        f"{model_id.replace('/', '_')}.json"
    )

    # ---- cache: skip reruns so we don't re-spend API/GPU -------------------
    if not force:
        results_volume.reload()
        if os.path.exists(path):
            with open(path) as f:
                cached = json.load(f)
            cached["cached"] = True
            return cached

    # ---- load the KernelBench problem --------------------------------------
    ds = load_dataset("ScalingIntelligence/KernelBench", split=f"level_{level}")
    problem = next(p for p in ds if p["problem_id"] == problem_num)
    code, name = problem["code"], problem["name"]

    # ---- query the model for a kernel --------------------------------------
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": build_prompt(code)}],
        temperature=0,
        # reasoning models (R1) spend many tokens thinking before the code;
        # a generous cap avoids truncated / empty completions (content=None).
        max_tokens=16000,
    )
    raw_response = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason

    # Reasoning models occasionally return content=None (e.g. the completion was
    # truncated mid-reasoning, finish_reason="length", or content-filtered). No
    # kernel was produced, so this is a generation failure — a distinct `empty`
    # category, not a syntactic one — and must not crash the run.
    if not raw_response or not raw_response.strip():
        category, detail, extra = "empty", f"no content (finish={finish_reason})", {}
    else:
        generated_code = _strip_fences(raw_response)
        category, detail, extra = classify(
            generated_code, code, use_sanitizer=use_sanitizer
        )

    result = {
        "problem_name": name,
        "level": level,
        "problem_id": problem_num,
        "model_id": model_id,
        "category": category,
        "detail": detail,
        "finish_reason": finish_reason,
        "raw_response": raw_response,
        "cached": False,
        **extra,
    }

    # ---- persist (cache: skip reruns later) --------------------------------
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()
    return result


def _silent_battery(ref_model, kernel_model, fresh_inputs, init_inputs, Model):
    """Return (core_failures, robustness_failures).

    Fixed, small, defensible battery (do not let this sprawl):
      CORE (same dtype, fp32 — a correct kernel MUST pass these -> `silent`):
        - fresh_seed     : 3 new random input draws (catches luck / hardcoding)
        - non_contig     : non-contiguous inputs (catches stride assumptions)
        - extreme        : large-magnitude inputs (catches overflow assumptions)
        - nondeterministic : same input twice must agree with ITSELF — a
                             reference-free metamorphic check that catches
                             uninitialised output / missing __syncthreads /
                             read-write races producing unstable results
      ROBUSTNESS (reported separately, NOT counted as silent):
        - fp16           : half precision (catches dtype assumptions)
    """
    import torch

    core = []
    robust = []

    def agree(inputs, ref_m, ker_m, atol=1e-3, rtol=1e-3):
        with torch.no_grad():
            r = ref_m(*inputs)
            k = ker_m(*inputs)
        return torch.allclose(r.float(), k.float(), atol=atol, rtol=rtol)

    # 1) fresh random draws — a correct kernel must pass every time
    try:
        for _ in range(3):
            if not agree(fresh_inputs(), ref_model, kernel_model):
                core.append("fresh_seed")
                break
    except Exception:
        core.append("fresh_seed_crash")

    # 2) non-contiguous inputs (same shape & values, broken strides)
    def make_noncontig(inputs):
        out = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                # identical values, broken strides
                x = torch.stack([x, x], dim=-1)[..., 0]
            out.append(x)
        return out
    try:
        if not agree(make_noncontig(fresh_inputs()), ref_model, kernel_model):
            core.append("non_contig")
    except Exception:
        core.append("non_contig_crash")

    # 3) nondeterminism — the kernel must agree with itself on identical input.
    # Reference-free: compares two kernel runs, so it flags unstable output even
    # where the kernel happens to match the reference on the first draw.
    try:
        det_inputs = fresh_inputs()
        with torch.no_grad():
            k1 = kernel_model(*det_inputs)
            k2 = kernel_model(*det_inputs)
        if not torch.allclose(k1.float(), k2.float(), atol=1e-3, rtol=1e-3):
            core.append("nondeterministic")
    except Exception:
        core.append("nondeterministic_crash")

    # 4) fp16 — both models cast to half, tolerance relaxed for precision
    try:
        ref_h = Model(*init_inputs).cuda().half().eval()
        ker_h = kernel_model.half()
        half_inputs = [
            x.half() if isinstance(x, torch.Tensor) else x for x in fresh_inputs()
        ]
        if not agree(half_inputs, ref_h, ker_h, atol=1e-2, rtol=1e-2):
            robust.append("fp16")
        kernel_model.float()  # restore for any later checks
    except Exception:
        robust.append("fp16_crash")

    # 5) extreme magnitude inputs (still fp32 -> core)
    def scale(inputs, c=100.0):
        return [x * c if isinstance(x, torch.Tensor) else x for x in inputs]
    try:
        if not agree(scale(fresh_inputs()), ref_model, kernel_model):
            core.append("extreme")
    except Exception:
        core.append("extreme_crash")

    return core, robust


def _parse_problems(spec: str):
    """Expand a problem spec like "1,2,5-8" into a sorted list of ints."""
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
    force: bool = False,
    use_sanitizer: bool = True,
):
    import itertools
    from collections import Counter

    models = {"v3": "deepseek/deepseek-chat", "r1": "deepseek/deepseek-r1"}
    nums = _parse_problems(problems)

    # fan out all (problem, model) runs across Modal containers at once
    handles = {}
    for num, (tag, model_id) in itertools.product(nums, models.items()):
        handles[(num, tag)] = diagnose.spawn(
            num, model_id, level, force, use_sanitizer
        )

    rows = []
    per_model = {tag: Counter() for tag in models}
    for (num, tag), h in handles.items():
        # a single container failing (infra hiccup, timeout) must not sink the
        # whole aggregation — record it as `error` and keep going.
        try:
            r = h.get()
        except Exception as e:
            r = {"category": "error", "detail": str(e)[:80]}
        rows.append((num, tag, r))
        per_model[tag][r["category"]] += 1

    rows.sort(key=lambda x: (x[0], x[1]))
    print(f"\n=== Level {level} — problems {nums} ===")
    for num, tag, r in rows:
        cache = " (cached)" if r.get("cached") else ""
        san = r.get("sanitizer_errors")
        san_s = f" san={san}" if san else ""
        detail = (r.get("detail") or "")[:60]
        print(
            f"L{level} p{num:<3} [{tag}] {r['category']:<10} "
            f"{detail}{san_s}{cache}"
        )

    print("\n--- category distribution (by model) ---")
    # canonical taxonomy first, then any extras actually seen (empty / error)
    canonical = ["syntactic", "runtime", "functional", "silent", "pass"]
    seen = set().union(*[set(c) for c in per_model.values()])
    cats = canonical + sorted(seen - set(canonical))
    header = "model  " + "".join(f"{c:>11}" for c in cats)
    print(header)
    for tag in models:
        line = f"{tag:<6} " + "".join(f"{per_model[tag].get(c, 0):>11}" for c in cats)
        print(line)

    # RQ2 headline: of the kernels that pass the standard allclose (silent+pass),
    # what fraction are actually silent failures?
    print("\n--- false-pass rate (RQ2) ---")
    for tag in models:
        silent = per_model[tag].get("silent", 0)
        passed = per_model[tag].get("pass", 0)
        denom = silent + passed
        if denom:
            pct = 100 * silent / denom
            print(
                f"{tag}: {silent}/{denom} = {pct:.0f}% of standard-passing "
                "kernels are silent failures"
            )
        else:
            print(f"{tag}: no standard-passing kernels")
