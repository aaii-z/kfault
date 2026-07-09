"""Official KernelBench one-shot CUDA/fp32 prompt, plus response parsing.

Matches src/kernelbench/prompt_constructor_toml.py (backend=cuda,
option=one_shot, precision=fp32) so results stay comparable to published
KernelBench pass rates. Built by concatenation, not str.format, so CUDA
braces are never parsed as format fields.
"""

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


def strip_reasoning(text: str) -> str:
    """Drop the <reasoning>...</reasoning> block Bedrock V3.2 puts in the
    message content. The thinking text can contain code fences, so this must
    run BEFORE strip_fences. An unclosed block means the completion was
    truncated mid-thought: nothing usable remains."""
    if "</reasoning>" in text:
        return text.split("</reasoning>")[-1].strip()
    if "<reasoning>" in text:
        return text.split("<reasoning>")[0].strip()
    return text.strip()


def strip_fences(text: str) -> str:
    """Extract the answer's code block, like KernelBench's extract_first_code.

    Prefers the segment that defines ModelNew: Bedrock V3.2 sometimes splits
    the opening fence around the </reasoning> tag, which leaves the code
    BEFORE the first remaining fence instead of after it. Drops a leading
    language tag line."""
    text = text.strip()
    if "```" not in text:
        return text
    segments = text.split("```")
    pick = None
    for seg in segments:
        if "class ModelNew" in seg:
            pick = seg
            break
    if pick is None:
        pick = segments[1] if len(segments) > 1 else segments[0]
    first_line, _, rest = pick.partition("\n")
    if first_line.strip().isalpha():
        pick = rest
    return pick.strip()
