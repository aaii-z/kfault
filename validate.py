"""T5: validation harness for the kfault classifier.

Feeds hand-crafted kernels of known category through the same classify()
the real pipeline uses, and asserts each label. Each case runs in its own
Modal container: a genuine runtime failure corrupts the CUDA context.

Run with: uv run modal run validate.py
"""

import modal

from kfault.classifier import classify
from pipeline import GPU, image

app = modal.App("kfault-validate")

# make kfault and pipeline importable inside the container
image = image.add_local_python_source("kfault", "pipeline")

# tiny reference problem: element-wise add, contiguous standard inputs
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
    """Element-wise add kernel around `add_body`; optionally .contiguous()."""
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


# weight-sync regression case: reference with learnable params. The generated
# model rebuilds the same nn.Linear and pushes its output through a correct
# CUDA copy kernel. Without seeding in classify() the two models get
# different weights and this correct kernel is mislabelled functional.
PROBLEM_LINEAR = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 256)

    def forward(self, x):
        return self.linear(x)


def get_inputs():
    return [torch.randn(64, 256)]


def get_init_inputs():
    return []
"""

KERNEL_LINEAR = '''
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void copy_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i];
}

torch::Tensor copy_cuda(torch::Tensor x) {
    x = x.contiguous();
    auto out = torch::empty_like(x);
    int n = x.numel();
    int block = 256;
    int grid = (n + block - 1) / block;
    copy_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

_mod = load_inline(
    name="kfault_val_linear_copy",
    cpp_sources="torch::Tensor copy_cuda(torch::Tensor x);",
    cuda_sources=_src,
    functions=["copy_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 256)

    def forward(self, x):
        return _mod.copy_cuda(self.linear(x))
'''


# name -> (expected category, problem, kernel)
CASES = {
    # correct and contiguity-safe
    "pass": ("pass", PROBLEM, _kernel("out[i] = a[i] + b[i];", contiguous=True)),
    # correct kernel on a problem with learnable weights
    "pass_weights": ("pass", PROBLEM_LINEAR, KERNEL_LINEAR),
    # `floatt` is not a type: fails to compile
    "syntactic": (
        "syntactic",
        PROBLEM,
        _kernel("floatt bad = a[i]; out[i] = a[i] + b[i];", contiguous=True),
    ),
    # null-pointer write: illegal memory access
    "runtime": (
        "runtime",
        PROBLEM,
        _kernel("int* p = nullptr; *p = i; out[i] = a[i] + b[i];", contiguous=True),
    ),
    # multiplies instead of adds: wrong on the standard input
    "functional": (
        "functional",
        PROBLEM,
        _kernel("out[i] = a[i] * b[i];", contiguous=True),
    ),
    # correct on contiguous input, but assumes contiguity
    "silent": (
        "silent",
        PROBLEM,
        _kernel("out[i] = a[i] + b[i];", contiguous=False),
    ),
}


@app.function(image=image, gpu=GPU, timeout=600)
def check_case(name: str, problem_code: str, kernel_code: str):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    category, detail, extra = classify(kernel_code, problem_code, use_sanitizer=True)
    return {"name": name, "category": category, "detail": detail, "extra": extra}


@app.local_entrypoint()
def main():
    handles = {
        name: check_case.spawn(name, problem, kernel)
        for name, (_, problem, kernel) in CASES.items()
    }

    print("\n=== T5 classifier validation ===")
    passed = 0
    for name, h in handles.items():
        expected = CASES[name][0]
        r = h.get()
        got = r["category"]
        ok = got == expected
        passed += ok
        mark = "OK " if ok else "XX "
        print(f"{mark} case={name:<13} expected={expected:<10} got={got:<10}")
        if not ok:
            print(f"      detail: {r['detail'][:1000]}")

    total = len(CASES)
    print(f"\n{passed}/{total} cases classified correctly")
    if passed != total:
        raise SystemExit("VALIDATION FAILED: classifier is not trustworthy yet")
    print("classifier validated")
