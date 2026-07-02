"""T5 — validation harness for the kfault classifier.

Feeds hand-crafted kernels of *known* failure category through the same
`classify()` the real pipeline uses, and asserts each is labelled correctly.
Without this, no number the pipeline produces is defensible.

Each case runs in its OWN Modal container: a genuine `runtime` failure (illegal
memory access) corrupts the CUDA context for the rest of the process, so the
cases must not share one. Run with:  `uv run modal run validate.py`
"""

import modal

from pipeline import classify, image

app = modal.App("kfault-validate")

# make the shared classifier importable inside the container
image = image.add_local_python_source("pipeline")

# ---- a tiny, self-contained reference problem (element-wise add) ------------
# Standard inputs are contiguous, so a kernel that silently assumes contiguity
# passes Stage 3 but is caught by the non-contiguous battery check.
PROBLEM = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    return [a, b]


def get_init_inputs():
    return []
"""


def _kernel(add_body: str, *, contiguous: bool) -> str:
    """Build a ModelNew that wraps `add_body` (the per-element expression) in an
    element-wise add kernel. `contiguous=True` calls .contiguous() first."""
    contig = "a = a.contiguous(); b = b.contiguous();" if contiguous else ""
    src = f"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{
        {add_body}
    }}
}}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {{
    {contig}
    auto out = torch::empty_like(a);
    int n = a.numel();
    int block = 256;
    int grid = (n + block - 1) / block;
    add_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}}
"""
    return f'''
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_src = r"""{src}"""

_mod = load_inline(
    name="kfault_val_{abs(hash((add_body, contiguous))) % 100000}",
    cpp_sources="torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);",
    cuda_sources=_src,
    functions=["add_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return _mod.add_cuda(a, b)
'''


# ---- the known-category kernels --------------------------------------------
CASES = {
    # correct AND contiguity-safe -> survives the whole battery
    "pass": _kernel("out[i] = a[i] + b[i];", contiguous=True),
    # won't compile: `floatt` is not a type -> Stage 1
    "syntactic": _kernel("floatt bad = a[i]; out[i] = a[i] + b[i];", contiguous=True),
    # null-pointer write in the kernel -> illegal memory access -> Stage 2
    "runtime": _kernel(
        "int* p = nullptr; *p = i; out[i] = a[i] + b[i];", contiguous=True
    ),
    # multiplies instead of adds -> wrong on the standard input -> Stage 3
    "functional": _kernel("out[i] = a[i] * b[i];", contiguous=True),
    # correct on contiguous std input, but assumes contiguity -> Stage 4 silent
    "silent": _kernel("out[i] = a[i] + b[i];", contiguous=False),
}


@app.function(image=image, gpu="T4", timeout=600)
def check_case(name: str, kernel_code: str):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    category, detail, extra = classify(kernel_code, PROBLEM, use_sanitizer=True)
    return {"name": name, "category": category, "detail": detail, "extra": extra}


@app.local_entrypoint()
def main():
    handles = {name: check_case.spawn(name, code) for name, code in CASES.items()}

    print("\n=== T5 classifier validation ===")
    passed = 0
    for name, h in handles.items():
        r = h.get()
        got = r["category"]
        ok = got == name
        passed += ok
        mark = "OK " if ok else "XX "
        print(f"{mark} expected={name:<10} got={got:<10}")
        if not ok:
            print(f"      detail: {r['detail'][:1000]}")

    total = len(CASES)
    print(f"\n{passed}/{total} cases classified correctly")
    if passed != total:
        raise SystemExit("VALIDATION FAILED — classifier is not trustworthy yet")
    print("classifier validated ✓")
