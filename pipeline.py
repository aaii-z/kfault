"""kfault diagnostic pipeline: classify an LLM-generated CUDA kernel into a
failure category — syntactic / runtime / functional / silent / pass.

This is the core of the dissertation. `demo.py` stays as the minimal reference;
this file is the evolving pipeline.

Categories (first matching stage wins):
  syntactic  - generated code fails to compile / build (nvcc or Python error)
  runtime    - compiles, but crashes when executed on standard input
  functional - runs, but wrong output on the standard input (basic allclose)
  silent     - passes the standard check, but WRONG under the stress battery
  pass       - correct on standard input AND survives the whole battery
"""

import modal

app = modal.App("kfault-pipeline")

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


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("openrouter")],
    volumes={RESULTS_DIR: results_volume},
    timeout=900,
)
def diagnose(
    problem_num: int = 1,
    model_id: str = "deepseek/deepseek-chat",
    level: int = 1,
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

    # ---- load the KernelBench problem & build the reference model ----------
    ds = load_dataset("ScalingIntelligence/KernelBench", split=f"level_{level}")
    problem = next(p for p in ds if p["problem_id"] == problem_num)
    code, name = problem["code"], problem["name"]

    problem_ns: dict = {}
    exec(code, problem_ns)
    Model = problem_ns["Model"]
    get_init_inputs = problem_ns.get("get_init_inputs", lambda: [])
    get_inputs = problem_ns.get("get_inputs", lambda: [])
    init_inputs = get_init_inputs()

    def fresh_inputs():
        """A new (randomised) set of inputs moved to CUDA."""
        return [
            x.cuda() if isinstance(x, torch.Tensor) else x
            for x in get_inputs()
        ]

    ref_model = Model(*init_inputs).cuda().eval()

    # ---- query the model for a kernel --------------------------------------
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": build_prompt(code)}],
        temperature=0,
    )
    raw_response = response.choices[0].message.content
    generated_code = _strip_fences(raw_response)

    result = {
        "problem_name": name,
        "level": level,
        "problem_id": problem_num,
        "model_id": model_id,
        "category": None,
        "detail": None,
        "raw_response": raw_response,
    }

    def _save(r):
        """Persist one run to the results Volume (cache: skip reruns later)."""
        path = f"{RESULTS_DIR}/level{level}_p{problem_num}_{model_id.replace('/', '_')}.json"
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        results_volume.commit()
        return r

    # ---- Stage 1: SYNTACTIC — does the generated code build? ---------------
    kernel_ns: dict = {}
    try:
        exec(generated_code, kernel_ns)          # load_inline compiles here
        ModelNew = kernel_ns["ModelNew"]
        kernel_model = ModelNew(*init_inputs).cuda().eval()
    except Exception as e:
        result["category"] = "syntactic"
        result["detail"] = str(e)[:500]
        return _save(result)

    # ---- Stage 2: RUNTIME — does it crash on standard input? ---------------
    std_inputs = fresh_inputs()
    try:
        with torch.no_grad():
            kernel_out = kernel_model(*std_inputs)
            ref_out = ref_model(*std_inputs)
        torch.cuda.synchronize()  # surface async kernel faults (illegal mem) here
    except Exception as e:
        result["category"] = "runtime"
        result["detail"] = str(e)[:500]
        return _save(result)

    # ---- Stage 3: FUNCTIONAL — correct on the standard input? --------------
    # a wrong-shaped output makes allclose raise; that is still a functional
    # failure (correct execution, wrong result), not a runtime crash.
    if ref_out.shape != kernel_out.shape:
        result["category"] = "functional"
        result["detail"] = f"shape mismatch {tuple(kernel_out.shape)} vs {tuple(ref_out.shape)}"
        return _save(result)
    if not torch.allclose(ref_out.float(), kernel_out.float(), atol=1e-3, rtol=1e-3):
        result["category"] = "functional"
        result["detail"] = "mismatch on standard input"
        return _save(result)

    # ---- Stage 4: SILENT — wrong under the fixed stress battery? -----------
    # core   = same-dtype (fp32) checks a correct kernel MUST pass -> defines `silent`
    # robust = fp16 etc., reported separately (NOT counted as silent, since
    #          KernelBench is fp32-only and flagging fp16 would inflate the number)
    core, robust = _silent_battery(ref_model, kernel_model, fresh_inputs, init_inputs, Model)
    result["robustness_flags"] = robust
    if core:
        result["category"] = "silent"
        result["detail"] = ", ".join(core)
        return _save(result)

    result["category"] = "pass"
    result["detail"] = "survived core battery" + (
        f" (robustness flags: {', '.join(robust)})" if robust else ""
    )
    return result


def _silent_battery(ref_model, kernel_model, fresh_inputs, init_inputs, Model):
    """Return (core_failures, robustness_failures).

    Fixed, small, defensible battery (do not let this sprawl):
      CORE (same dtype, fp32 — a correct kernel MUST pass these -> `silent`):
        - fresh_seed   : 3 new random input draws (catches luck / hardcoding)
        - non_contig   : non-contiguous inputs (catches stride assumptions)
        - extreme      : large-magnitude inputs (catches overflow assumptions)
      ROBUSTNESS (reported separately, NOT counted as silent):
        - fp16         : half precision (catches dtype assumptions)
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
                x = torch.stack([x, x], dim=-1)[..., 0]  # identical values, non-contiguous
            out.append(x)
        return out
    try:
        if not agree(make_noncontig(fresh_inputs()), ref_model, kernel_model):
            core.append("non_contig")
    except Exception:
        core.append("non_contig_crash")

    # 3) fp16 — both models cast to half, tolerance relaxed for precision
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

    # 4) extreme magnitude inputs (still fp32 -> core)
    def scale(inputs, c=100.0):
        return [x * c if isinstance(x, torch.Tensor) else x for x in inputs]
    try:
        if not agree(scale(fresh_inputs()), ref_model, kernel_model):
            core.append("extreme")
    except Exception:
        core.append("extreme_crash")

    return core, robust


@app.local_entrypoint()
def main(level: int = 1, problems: str = "1"):
    import itertools

    models = {"v3": "deepseek/deepseek-chat", "r1": "deepseek/deepseek-r1"}
    nums = [int(p) for p in problems.split(",")]

    for num, (tag, model_id) in itertools.product(nums, models.items()):
        r = diagnose.remote(num, model_id, level)
        print(f"\nL{level} {r['problem_name']} [{tag}]")
        print(f"  category : {r['category']}")
        print(f"  detail   : {r['detail']}")
